# AlphaFold 3 Decoded
This repository guides you through doing a full implementation on AlphaFold 3 yourself. I'm creating an accompanying YouTube series that will be released soon. 

## Setup
Setup is a bit more complex this time around. To prepare you for all the possible things you want to do within this repo, including potential cloud training on remote instances, I recommend you go through the following steps:
- Setup SSH access to GitHub: Explained on the [GitHub manual pages](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). This allows you to make modifications to your GitHub repo from remote machines, without having to copy your credentials to them.
- Fork this Repository: This way, you can use Git for version control yourself, which makes it easy to get your code from and to remote machines.
- Install the package manager uv. Instructions can be found on the [uv website](https://docs.astral.sh/uv/getting-started/installation/).
- Clone this repository from your fork. Particularly if you are working on a remote machine, I suggest you use the SSH link from your GitHub page, so you can make commits via your SSH key.
- Install the dependencies: `uv sync`
- Activate the virtual environment: In your code editor, or manually using `source .venv/bin/activate`
- Compute the test results: `python scripts/generate_test_results.py`, select the chapter you want to generate results for. You have to do this on your machine, so that the random generators work correctly (PyTorch is not deterministic over different machines).
- Set the Jupyter Root: In VSCode, go to Settings > Jupyter: Notebook File Root, and set it to `${workspaceFolder}/tutorials`. This makes sure that the `tutorials` folder is on your python path, and that the local files in `data/` are available. 

> **Note**: If you already started the series before I completed it, you should add the original upstream repository to your Git:

```git remote add upstream https://github.com/kilianmandon/alphafold3-decoded.git```

This way, you can pull updates for the new chapters into your own repository, once I coded them:

```
# git stash if you made non-commited changes
git fetch upstream
git merge upstream/main
# git stash pop if you made non-commited changes
```

## Optional: Loading AlphaFold Weights
As a researcher, you can request the pretrained weights from Deepmind: https://github.com/google-deepmind/alphafold3

These weights are compatible with the model you are building. If you obtained them, you can place them in `data/params/af3.bin.zst` and run `python scripts/remap_weights.py` to convert them to the required format for this repo. This is only necessary if you want to run actual predictions.


## Working on the Tutorials
The Jupyter Notebooks contain a lot of information on their own, so if you are not a YouTube person, you could also try to do them without watching the video first. 

The notebooks tutorials/_chapter_/_chapter_.ipynb guide you through the implementation. All of your real coding is done in the .py files, but the notebooks contain info and can run the test cases. Alternatively, the folder `tutorials/converted_notebooks` contains the same notebooks in .py format. You can also run the test cases through them. This can be helpful if you want to use the Debugger, which is easier in python scripts compared to jupyter notebooks.

Basically, the python files will contain sections in the form of 
```
""" TODO: Implement this function. These are some hints on how to do it: ... """

# Replace 'pass' with your code
pass

""" End of your code """
```
and you will be replacing 'pass' with your code. 

## Current State
| Chapter | Code | Video |
|---|---|---|
| feature_extraction | ✅ Ready | 🎬 In Progress |
| input_embedding | — | — |
| evoformer | — | — |
| diffusion | — | — |
| training | — | — |

Note that Ready doesn't mean rigorously tested :)