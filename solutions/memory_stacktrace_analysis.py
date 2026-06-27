import ast
import re
import rich
import tqdm
import pickle

import numpy as np


def process_traces(source_code, trace_events, filename_whitelist=None):

    node_by_lineno = {
        f: {} for f in source_code
    }

    print('Parsing source code...')
    for (filename, code) in source_code.items():
        src_tree = ast.parse(code, filename)
        for node in ast.walk(src_tree):
            if isinstance(node, ast.ClassDef):
                node_by_lineno[filename][(node.lineno, node.end_lineno)] = node

    def extract_context_data(node, lineno):
        splits = []
        splits.append((node.name, node.end_lineno))

        for child_node in ast.walk(node):
            if isinstance(child_node, ast.ClassDef) or isinstance(child_node, ast.FunctionDef):
                if child_node.lineno <= lineno < child_node.end_lineno:
                    splits.append((child_node.name, child_node.end_lineno))
        splits = sorted(splits, key=lambda x: x[1])
        
        context_str = '.'.join(x[0] for x in splits)
        context_data = {
            'context_str': context_str,
            'context_str_lineno': f'{context_str}:{lineno}',
            'lineno': lineno,
            'source_node': node,
        }

        return context_data

    context_quick_lookup = {}
    def get_context(filename, lineno):
        lookup_key = f'{filename}:{lineno}'
        if lookup_key in context_quick_lookup:
            return context_quick_lookup[lookup_key]

        for (line_range, node) in node_by_lineno[filename].items():
            if line_range[0] <= lineno < line_range[1]:
                context_data = extract_context_data(node, lineno)
                context_quick_lookup[lookup_key] = context_data
                return context_data

        context_quick_lookup[lookup_key] = None
        return None
                
    print('Annotating trace events...')
    for trace_event in tqdm.tqdm(trace_events):
        trace_event_contexts = []
        trace_event_frames = []

        if 'forward_frames' in trace_event:
            regex = re.compile(r'\s*File "(.*?)", line (\d+)')

            for frame in trace_event['forward_frames']:
                match = regex.search(frame)
                filename = match.group(1)
                line = int(match.group(2))
                trace_event_frames.insert(0, {
                    'filename': filename,
                    'line': line
                })
        else:
            trace_event_frames += trace_event['frames']

        for frame in trace_event_frames:
            if frame['filename'] not in node_by_lineno:
                continue
            if filename_whitelist is not None and not filename_whitelist(frame['filename']):
                continue

            context = get_context(frame['filename'], frame['line'])
            if context is None:
                continue

            trace_event_contexts.append(context)
        trace_event['context_stack'] = list(reversed(trace_event_contexts))

def get_peak_events(memory_snapshot_data):
    trace_events = memory_snapshot_data['device_traces'][0]
    num_allocs = len([a for a in trace_events if a['action'] == 'alloc'])
    interval_ids = np.zeros((num_allocs,), dtype=int)
    intervals = np.zeros((num_allocs, 2), dtype=int)
    intervals[:, 1] = int(1e10)
    amounts = np.zeros((num_allocs,), dtype=int)
    memory_addr_to_index_map = {}

    current_counter = 0

    for i, trace_event in enumerate(trace_events):
        if trace_event['action'] == 'alloc':
            if trace_event['addr'] in memory_addr_to_index_map:
                # Check that address was freed before reuse
                last_index = memory_addr_to_index_map[trace_event['addr']]
                assert intervals[last_index, 1] < 1e10, 'Address reused before it was freed'

            memory_addr_to_index_map[trace_event['addr']] = current_counter
            interval_ids[current_counter] = trace_event['addr']
            intervals[current_counter, 0] = i
            amounts[current_counter] = trace_event['size']
            current_counter += 1
        elif trace_event['action'] == 'free_completed':
            if trace_event['addr'] not in memory_addr_to_index_map:
                raise ValueError('Freeing before allocation')
            counter = memory_addr_to_index_map[trace_event['addr']]
            intervals[counter, 1] = i

    memory_changes = np.zeros((len(trace_events),), dtype=int)
    for amount, interval in zip(amounts, intervals):
        memory_changes[interval[0]] += amount

        if interval[1] < memory_changes.shape[0]:
            # This is False for events that are not ended
            memory_changes[interval[1]] -= amount
    
    cumulative_memory = np.cumsum(memory_changes)
    peak_index = np.argmax(cumulative_memory)
    peak_memory = cumulative_memory[peak_index]

    participating_inds = np.nonzero((intervals[:, 0] <= peak_index) & (peak_index < intervals[:, 1]))[0]
    participating_events = [trace_events[intervals[i, 0]] for i in participating_inds]

    return participating_events


def memory_analysis(memory_snapshot_file):
    with open(memory_snapshot_file, 'rb') as f:
        data = pickle.load(f)

    peak_events = get_peak_events(data)
    
    source_code = data['source_code']
    process_traces(source_code, peak_events, lambda f: 'solutions' in f)

    tree_data = { 'name': 'total' }
    for event in peak_events:
        current_level = tree_data
        for context in event['context_stack']:
            context_str = context['context_str_lineno']
            if context_str not in current_level:
                current_level[context_str] = { 'name': context_str, 'context': context }
            current_level = current_level[context_str]

        current_level['value'] = current_level.get('value', 0) + event['size']
        current_level['count'] = current_level.get('count', 0) + 1
        current_level_events = current_level.get('event_names', [])
        current_level_events.append(event)
        current_level['event_names'] = current_level_events

    def post_order_traversal(node, fn):
        for child in node.values():
            if not isinstance(child, dict):
                continue
            post_order_traversal(child, fn)
        fn(node)

    def sum_up_child_values(node):
        node['value'] = node.get('value', 0) + sum(a['value'] for a in node.values() if isinstance(a, dict))
        node['count'] = node.get('count', 0) + sum(a['count'] for a in node.values() if isinstance(a, dict))

    post_order_traversal(tree_data, sum_up_child_values)


    def preorder_print_value(node, depth):
        if depth == 1:
            color = 'dark_orange'
        elif depth == 2:
            color = 'dark_turquoise'
        elif depth == 3:
            color = 'light_green'
        elif depth == 4:
            color = 'chartreuse4'
        else:
            color = 'white'

        amount_gib = node['value'] / 1024**3
        if amount_gib > 0.5:
            to_print = (' ' * depth * 2) + f'[{color}]{node["name"]}:[/{color}] {amount_gib:.2f} GB ({node["count"]} events)'

            rich.print(to_print)
        for child in node.values():
            if not isinstance(child, dict):
                continue
            preorder_print_value(child, depth+1)

    preorder_print_value(tree_data, depth=0)
    # for ev in tree_data['event_names']:
    #     if ev['size'] > 1024**3:
    #         print(ev)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot_file')
    args = parser.parse_args()
    memory_analysis(args.snapshot_file)

if __name__=='__main__':
    main()
