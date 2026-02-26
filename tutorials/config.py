from dataclasses import dataclass, field

@dataclass
class GlobalConfig:
    # Single representation
    c_s: int = 384
    # MSA representation
    c_m: int = 64
    # Pair representation
    c_z: int = 128
    # target feature embedding, aka s_input
    c_s_input: int = 449
    # Number of recycling iterations
    n_cycle: int = 11
    # relative encoding feature
    rel_feat_dim: int = 139
    # MSA feature
    msa_feat_dim: int = 34

@dataclass
class FeaturizationConfig:
    max_msa_sequences: int = 16384
    msa_trunc_count: int = 1024

@dataclass
class AtomAttentionConfig:
    atom_element_dim: int = 128
    atom_chars_dim: int = 64
    c_atom: int = 128
    c_atompair: int = 16
    c_token: int = None
    n_head_atom_transformer: int = 4
    n_block_atom_transformer: int = 3

@dataclass
class InputEmbeddingConfig:
    atom_attention_config: AtomAttentionConfig = field(default_factory=lambda: AtomAttentionConfig(c_token=384))
    r_max: int = 32
    s_max: int = 2


@dataclass
class MSAModuleConfig:
    n_blocks: int = 4
    n_transition: int = 4
    p_dropout: float = 0.15
    n_transition_pairstack: int = 4
    p_dropout_pairstack: float = 0.25
    n_head_pairstack: int = 4
    c_opm: int = 32
    c_msa_ave: int = 8
    n_head_msa_ave: int =  8

@dataclass
class TemplateModuleConfig:
    c_in: int = 106
    c: int = 64
    n_blocks: int = 2
    n_templates: int = 4
    n_transition_pairstack: int = 2
    n_head_pairstack: int = 4
    p_dropout_pairstack: float = 0.25


@dataclass
class PairformerConfig:
    n_blocks: int = 48
    n_transition: int = 4
    n_head_att_pair_bias: int = 16
    n_head_pairstack: int = 4
    n_transition_pairstack: int = 4
    p_dropout_pairstack: float = 0.25

@dataclass
class EvoformerConfig:
    msa_module_config: MSAModuleConfig = field(default_factory=lambda: MSAModuleConfig())
    template_module_config: TemplateModuleConfig = field(default_factory=lambda: TemplateModuleConfig())
    pairformer_config: PairformerConfig = field(default_factory=lambda: PairformerConfig())



@dataclass
class DiffusionConfig:
    sigma_data: float = 16.0

    # Augmentation
    s_trans_center_randaug: float = 1.0

    # Atom Attention Encoder
    atom_attention_config: AtomAttentionConfig = field(default_factory=lambda: AtomAttentionConfig(c_token=768))

    # Positional Embeddings
    c_fourier: int = 256

    # Diffusion Transformer
    n_block_diffusion_transformer: int = 24
    n_head_diffusion_transformer: int = 16

    # Diffusion Sampler
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    denoising_steps: int = 30

    # Noise Schedule
    s_min: float = 0.0004
    s_max: float = 160.0
    rho: int = 7


@dataclass
class Config:
    global_config: GlobalConfig = field(default_factory=lambda: GlobalConfig())
    featurization_config: FeaturizationConfig = field(default_factory=lambda: FeaturizationConfig())
    input_embedding_config: InputEmbeddingConfig = field(default_factory=lambda: InputEmbeddingConfig())
    evoformer_config: EvoformerConfig = field(default_factory=lambda: EvoformerConfig())
    diffusion_config: DiffusionConfig = field(default_factory=lambda: DiffusionConfig())
