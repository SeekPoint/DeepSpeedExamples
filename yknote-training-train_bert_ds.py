(yk_py39) amd00@MZ32-00:~/yk_fork/DeepSpeedExamples/training/HelloDeepSpeed$ git diff train_bert_ds.py
diff --git a/training/HelloDeepSpeed/train_bert_ds.py b/training/HelloDeepSpeed/train_bert_ds.py
index 98f43fc..78c1281 100644
--- a/training/HelloDeepSpeed/train_bert_ds.py
+++ b/training/HelloDeepSpeed/train_bert_ds.py
@@ -31,6 +31,7 @@ from transformers.models.roberta.modeling_roberta import (
     RobertaPreTrainedModel,
 )
 
+os.environ['CUDA_VISIBLE_DEVICES'] ='0'
 
 def is_rank_0() -> bool:
     return int(os.environ.get("RANK", "0")) == 0
@@ -263,7 +264,7 @@ def create_data_iterator(
             be able to continuously generate samples
 
     """
-    wikitext_dataset = datasets.load_dataset("wikitext",
+    wikitext_dataset = datasets.load_dataset("/home/amd00/hf_model/wikitext",
                                              "wikitext-2-v1",
                                              split="train")
     wikitext_dataset = wikitext_dataset.filter(
@@ -608,8 +609,8 @@ def train(
         dropout: float = 0.1,
         # Training Parameters
         batch_size: int = 8,
-        num_iterations: int = 10000,
-        checkpoint_every: int = 1000,
+        num_iterations: int = 50,
+        checkpoint_every: int = 10,
         log_every: int = 10,
         local_rank: int = -1,
 ) -> pathlib.Path:
@@ -667,8 +668,9 @@ def train(
         pathlib.Path: The final experiment directory
 
     """
-    device = (torch.device("cuda", local_rank) if (local_rank > -1)
-              and torch.cuda.is_available() else torch.device("cpu"))
+    device = torch.device("cuda", 0) 
+    #device = (torch.device("cuda", 0) if (local_rank > -1)
+    #          and torch.cuda.is_available() else torch.device("cpu"))
     ################################
     ###### Create Exp. Dir #########
     ################################
(yk_py39) amd00@MZ32-00:~/yk_fork/DeepSpeedExamples/training/HelloDeepSpeed$ python train_bert_ds.py --checkpoint_dir ./experiments
[2023-08-02 00:41:06,181] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
2023-08-02 00:41:07.043 | INFO     | __main__:log_dist:54 - [Rank 0] Creating Experiment Directory
2023-08-02 00:41:07.128 | INFO     | __main__:log_dist:54 - [Rank 0] Experiment Directory created at experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg
2023-08-02 00:41:07.130 | INFO     | __main__:log_dist:54 - [Rank 0] Creating Datasets
Reusing dataset wikitext (/home/amd00/.cache/huggingface/datasets/wikitext/wikitext-2-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126)
Loading cached processed dataset at /home/amd00/.cache/huggingface/datasets/wikitext/wikitext-2-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/cache-43b54ffba72698be.arrow
Parameter 'function'=<function create_data_iterator.<locals>.<lambda> at 0x7f241d6f1700> of the transform datasets.arrow_dataset.Dataset._map_single couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
Loading cached processed dataset at /home/amd00/.cache/huggingface/datasets/wikitext/wikitext-2-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126/cache-629f6fbed82c07cd.arrow
2023-08-02 00:41:09.317 | INFO     | __main__:log_dist:54 - [Rank 0] Dataset Creation Done
2023-08-02 00:41:09.317 | INFO     | __main__:log_dist:54 - [Rank 0] Creating Model
2023-08-02 00:41:09.687 | INFO     | __main__:log_dist:54 - [Rank 0] Model Creation Done
2023-08-02 00:41:09.687 | INFO     | __main__:log_dist:54 - [Rank 0] Creating DeepSpeed engine
[2023-08-02 00:41:09,687] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.10.0+f5c834a6, git-hash=f5c834a6, git-branch=tag_v0.10.0
[2023-08-02 00:41:09,687] [WARNING] [comm.py:152:init_deepspeed_backend] NCCL backend in DeepSpeed not yet implemented
[2023-08-02 00:41:09,688] [INFO] [comm.py:616:init_distributed] cdb=None
[2023-08-02 00:41:09,688] [INFO] [comm.py:627:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2023-08-02 00:41:09,815] [INFO] [comm.py:677:mpi_discovery] Discovered MPI settings of world_rank=0, local_rank=0, world_size=1, master_addr=192.168.1.11, master_port=29500
[2023-08-02 00:41:09,815] [INFO] [comm.py:643:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
[2023-08-02 00:41:11,046] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2023-08-02 00:41:12,283] [WARNING] [cpu_adam.py:84:__init__] FP16 params for CPUAdam may not work on AMD CPUs
Using /home/amd00/.cache/torch_extensions/py39_cu117 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/amd00/.cache/torch_extensions/py39_cu117/cpu_adam/build.ninja...
Building extension module cpu_adam...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module cpu_adam...
Time to load cpu_adam op: 2.844942569732666 seconds
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.000100, betas=(0.900000, 0.999000), weight_decay=0.000000, adam_w=1
[2023-08-02 00:41:16,834] [INFO] [logging.py:96:log_dist] [Rank 0] Using DeepSpeed Optimizer param name adam as basic optimizer
[2023-08-02 00:41:16,837] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Basic Optimizer = DeepSpeedCPUAdam
[2023-08-02 00:41:16,837] [INFO] [utils.py:54:is_zero_supported_optimizer] Checking ZeRO support for optimizer=DeepSpeedCPUAdam type=<class 'deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam'>
[2023-08-02 00:41:16,837] [INFO] [logging.py:96:log_dist] [Rank 0] Creating torch.float16 ZeRO stage 1 optimizer
[2023-08-02 00:41:16,837] [INFO] [stage_1_and_2.py:133:__init__] Reduce bucket size 500,000,000
[2023-08-02 00:41:16,837] [INFO] [stage_1_and_2.py:134:__init__] Allgather bucket size 500,000,000
[2023-08-02 00:41:16,837] [INFO] [stage_1_and_2.py:135:__init__] CPU Offload: True
[2023-08-02 00:41:16,837] [INFO] [stage_1_and_2.py:136:__init__] Round robin gradient partitioning: False
Rank: 0 partition count [1] and sizes[(16345178, False)] 
[2023-08-02 00:41:17,097] [INFO] [utils.py:785:see_memory_usage] Before initializing optimizer states
[2023-08-02 00:41:17,098] [INFO] [utils.py:786:see_memory_usage] MA 0.06 GB         Max_MA 0.06 GB         CA 0.06 GB         Max_CA 0 GB 
[2023-08-02 00:41:17,098] [INFO] [utils.py:793:see_memory_usage] CPU Virtual Memory:  used = 9.24 GB, percent = 3.7%
[2023-08-02 00:41:17,255] [INFO] [utils.py:785:see_memory_usage] After initializing optimizer states
[2023-08-02 00:41:17,255] [INFO] [utils.py:786:see_memory_usage] MA 0.06 GB         Max_MA 0.06 GB         CA 0.06 GB         Max_CA 0 GB 
[2023-08-02 00:41:17,256] [INFO] [utils.py:793:see_memory_usage] CPU Virtual Memory:  used = 9.43 GB, percent = 3.7%
[2023-08-02 00:41:17,256] [INFO] [stage_1_and_2.py:493:__init__] optimizer state initialized
[2023-08-02 00:41:17,450] [INFO] [utils.py:785:see_memory_usage] After initializing ZeRO optimizer
[2023-08-02 00:41:17,451] [INFO] [utils.py:786:see_memory_usage] MA 0.06 GB         Max_MA 0.06 GB         CA 0.06 GB         Max_CA 0 GB 
[2023-08-02 00:41:17,451] [INFO] [utils.py:793:see_memory_usage] CPU Virtual Memory:  used = 9.43 GB, percent = 3.7%
[2023-08-02 00:41:17,455] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Final Optimizer = adam
[2023-08-02 00:41:17,455] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed using client LR scheduler
[2023-08-02 00:41:17,455] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed LR Scheduler = None
[2023-08-02 00:41:17,455] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[0.0001], mom=[(0.9, 0.999)]
[2023-08-02 00:41:17,455] [INFO] [config.py:960:print] DeepSpeedEngine configuration:
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   activation_checkpointing_config  {
    "partition_activations": false, 
    "contiguous_memory_optimization": false, 
    "cpu_checkpointing": false, 
    "number_checkpoints": null, 
    "synchronize_checkpoint_boundary": false, 
    "profile": false
}
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True}
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   amp_enabled .................. False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   amp_params ................... False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   autotuning_config ............ {
    "enabled": false, 
    "start_step": null, 
    "end_step": null, 
    "metric_path": null, 
    "arg_mappings": null, 
    "metric": "throughput", 
    "model_info": null, 
    "results_dir": "autotuning_results", 
    "exps_dir": "autotuning_exps", 
    "overwrite": true, 
    "fast": true, 
    "start_profile_step": 3, 
    "end_profile_step": 5, 
    "tuner_type": "gridsearch", 
    "tuner_early_stopping": 5, 
    "tuner_num_trials": 50, 
    "model_info_path": null, 
    "mp_size": 1, 
    "max_train_batch_size": null, 
    "min_train_batch_size": 1, 
    "max_train_micro_batch_size_per_gpu": 1.024000e+03, 
    "min_train_micro_batch_size_per_gpu": 1, 
    "num_tuning_micro_batch_sizes": 3
}
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   bfloat16_enabled ............. False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   checkpoint_parallel_write_pipeline  False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   checkpoint_tag_validation_enabled  True
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   checkpoint_tag_validation_fail  False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x7f2395cab1c0>
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   communication_data_type ...... None
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   curriculum_enabled_legacy .... False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   curriculum_params_legacy ..... False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   data_efficiency_enabled ...... False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   dataloader_drop_last ......... False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   disable_allgather ............ False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   dump_state ................... False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   dynamic_loss_scale_args ...... None
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   eigenvalue_enabled ........... False
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   eigenvalue_gas_boundary_resolution  1
[2023-08-02 00:41:17,456] [INFO] [config.py:964:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   eigenvalue_layer_num ......... 0
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   eigenvalue_max_iter .......... 100
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   eigenvalue_stability ......... 1e-06
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   eigenvalue_tol ............... 0.01
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   eigenvalue_verbose ........... False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   elasticity_enabled ........... False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   flops_profiler_config ........ {
    "enabled": false, 
    "recompute_fwd_factor": 0.0, 
    "profile_step": 1, 
    "module_depth": -1, 
    "top_modules": 1, 
    "detailed": true, 
    "output_file": null
}
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   fp16_auto_cast ............... False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   fp16_enabled ................. True
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   fp16_master_weights_and_gradients  False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   global_rank .................. 0
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   grad_accum_dtype ............. None
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   gradient_accumulation_steps .. 1
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   gradient_clipping ............ 0.0
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   gradient_predivide_factor .... 1.0
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   initial_dynamic_scale ........ 65536
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   load_universal_checkpoint .... False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   loss_scale ................... 0
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   memory_breakdown ............. False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   mics_hierarchial_params_gather  False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   mics_shard_size .............. -1
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') enabled=False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   nebula_config ................ {
    "enabled": false, 
    "persistent_storage_path": null, 
    "persistent_time_interval": 100, 
    "num_of_version_in_retention": 2, 
    "enable_nebula_load": true, 
    "load_path": null
}
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   optimizer_legacy_fusion ...... False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   optimizer_name ............... adam
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   optimizer_params ............. {'lr': 0.0001}
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0}
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   pld_enabled .................. False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   pld_params ................... False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   prescale_gradients ........... False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   scheduler_name ............... None
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   scheduler_params ............. None
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   sparse_attention ............. None
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   sparse_gradients_enabled ..... False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   steps_per_print .............. 10
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   train_batch_size ............. 8
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   train_micro_batch_size_per_gpu  8
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   use_node_local_storage ....... False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   wall_clock_breakdown ......... False
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   world_size ................... 1
[2023-08-02 00:41:17,457] [INFO] [config.py:964:print]   zero_allow_untested_optimizer  False
[2023-08-02 00:41:17,458] [INFO] [config.py:964:print]   zero_config .................. stage=1 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500,000,000 allgather_partitions=True allgather_bucket_size=500,000,000 overlap_comm=False load_from_fp32_weights=True elastic_checkpoint=False offload_param=None offload_optimizer=DeepSpeedZeroOffloadOptimizerConfig(device='cpu', nvme_path=None, buffer_count=4, pin_memory=False, pipeline=False, pipeline_read=False, pipeline_write=False, fast_init=False) sub_group_size=1,000,000,000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50,000,000 param_persistence_threshold=100,000 model_persistence_threshold=sys.maxsize max_live_parameters=1,000,000,000 max_reuse_distance=1,000,000,000 gather_16bit_weights_on_model_save=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True
[2023-08-02 00:41:17,458] [INFO] [config.py:964:print]   zero_enabled ................. True
[2023-08-02 00:41:17,458] [INFO] [config.py:964:print]   zero_force_ds_cpu_optimizer .. True
[2023-08-02 00:41:17,458] [INFO] [config.py:964:print]   zero_optimization_stage ...... 1
[2023-08-02 00:41:17,458] [INFO] [config.py:950:print_user_config]   json = {
    "train_micro_batch_size_per_gpu": 8, 
    "optimizer": {
        "type": "Adam", 
        "params": {
            "lr": 0.0001
        }
    }, 
    "fp16": {
        "enabled": true
    }, 
    "zero_optimization": {
        "stage": 1, 
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
2023-08-02 00:41:17.458 | INFO     | __main__:log_dist:54 - [Rank 0] DeepSpeed engine created
2023-08-02 00:41:17.458 | INFO     | __main__:log_dist:54 - [Rank 0] Total number of model parameters: 16,345,177
[2023-08-02 00:41:17,526] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4294967296, reducing to 2147483648
[2023-08-02 00:41:17,578] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 2147483648, reducing to 1073741824
[2023-08-02 00:41:17,623] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1073741824, reducing to 536870912
[2023-08-02 00:41:17,668] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 536870912, reducing to 268435456
[2023-08-02 00:41:17,712] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 268435456, reducing to 134217728
[2023-08-02 00:41:17,760] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 134217728, reducing to 67108864
[2023-08-02 00:41:17,804] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 67108864, reducing to 33554432
[2023-08-02 00:41:17,848] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 33554432, reducing to 16777216
[2023-08-02 00:41:17,891] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 16777216, reducing to 8388608
[2023-08-02 00:41:17,935] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 8388608, reducing to 4194304
[2023-08-02 00:41:17,935] [INFO] [logging.py:96:log_dist] [Rank 0] step=10, skipped=10, lr=[0.0001], mom=[(0.9, 0.999)]
[2023-08-02 00:41:17,936] [INFO] [timer.py:215:stop] epoch=0/micro_step=10/global_step=10, RunningAvgSamplesPerSec=220.69530887096045, CurrSamplesPerSec=222.88787331278564, MemAllocated=0.07GB, MaxMemAllocated=1.36GB
2023-08-02 00:41:17.936 | INFO     | __main__:log_dist:54 - [Rank 0] Loss: 10.8766
[2023-08-02 00:41:17,937] [INFO] [logging.py:96:log_dist] [Rank 0] [Torch] Checkpoint global_step10 is about to be saved!
/home/amd00/anaconda3/envs/yk_py39/lib/python3.9/site-packages/torch/nn/modules/module.py:1802: UserWarning: Positional args are being deprecated, use kwargs instead. Refer to https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict for details.
  warnings.warn(
[2023-08-02 00:41:17,939] [INFO] [logging.py:96:log_dist] [Rank 0] Saving model checkpoint: experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step10/mp_rank_00_model_states.pt
[2023-08-02 00:41:17,939] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step10/mp_rank_00_model_states.pt...
[2023-08-02 00:41:17,977] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step10/mp_rank_00_model_states.pt.
[2023-08-02 00:41:17,978] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step10/zero_pp_rank_0_mp_rank_00_optim_states.pt...
[2023-08-02 00:41:18,165] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step10/zero_pp_rank_0_mp_rank_00_optim_states.pt.
[2023-08-02 00:41:18,166] [INFO] [engine.py:3285:_save_zero_checkpoint] zero checkpoint saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step10/zero_pp_rank_0_mp_rank_00_optim_states.pt
[2023-08-02 00:41:18,166] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step10 is ready now!
2023-08-02 00:41:18.166 | INFO     | __main__:log_dist:54 - [Rank 0] Saved model to experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg
[2023-08-02 00:41:18,216] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4194304, reducing to 2097152
[2023-08-02 00:41:18,260] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 2097152, reducing to 1048576
[2023-08-02 00:41:18,305] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1048576, reducing to 524288
[2023-08-02 00:41:18,349] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 524288, reducing to 262144
[2023-08-02 00:41:18,394] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072
[2023-08-02 00:41:18,438] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 131072, reducing to 65536
[2023-08-02 00:41:18,482] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 65536, reducing to 32768
[2023-08-02 00:41:18,676] [INFO] [logging.py:96:log_dist] [Rank 0] step=20, skipped=17, lr=[0.0001], mom=[(0.9, 0.999)]
[2023-08-02 00:41:18,677] [INFO] [timer.py:215:stop] epoch=0/micro_step=20/global_step=20, RunningAvgSamplesPerSec=202.965811549343, CurrSamplesPerSec=139.10360295001638, MemAllocated=0.07GB, MaxMemAllocated=1.36GB
2023-08-02 00:41:18.677 | INFO     | __main__:log_dist:54 - [Rank 0] Loss: 10.8680
[2023-08-02 00:41:18,678] [INFO] [logging.py:96:log_dist] [Rank 0] [Torch] Checkpoint global_step20 is about to be saved!
[2023-08-02 00:41:18,680] [INFO] [logging.py:96:log_dist] [Rank 0] Saving model checkpoint: experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step20/mp_rank_00_model_states.pt
[2023-08-02 00:41:18,680] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step20/mp_rank_00_model_states.pt...
[2023-08-02 00:41:18,721] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step20/mp_rank_00_model_states.pt.
[2023-08-02 00:41:18,722] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step20/zero_pp_rank_0_mp_rank_00_optim_states.pt...
[2023-08-02 00:41:18,927] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step20/zero_pp_rank_0_mp_rank_00_optim_states.pt.
[2023-08-02 00:41:18,928] [INFO] [engine.py:3285:_save_zero_checkpoint] zero checkpoint saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step20/zero_pp_rank_0_mp_rank_00_optim_states.pt
[2023-08-02 00:41:18,928] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step20 is ready now!
2023-08-02 00:41:18.928 | INFO     | __main__:log_dist:54 - [Rank 0] Saved model to experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg
[2023-08-02 00:41:19,600] [INFO] [logging.py:96:log_dist] [Rank 0] step=30, skipped=17, lr=[0.0001], mom=[(0.9, 0.999)]
[2023-08-02 00:41:19,600] [INFO] [timer.py:215:stop] epoch=0/micro_step=30/global_step=30, RunningAvgSamplesPerSec=175.46016652134068, CurrSamplesPerSec=144.43879488095598, MemAllocated=0.07GB, MaxMemAllocated=1.36GB
2023-08-02 00:41:19.600 | INFO     | __main__:log_dist:54 - [Rank 0] Loss: 10.7828
[2023-08-02 00:41:19,601] [INFO] [logging.py:96:log_dist] [Rank 0] [Torch] Checkpoint global_step30 is about to be saved!
[2023-08-02 00:41:19,604] [INFO] [logging.py:96:log_dist] [Rank 0] Saving model checkpoint: experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step30/mp_rank_00_model_states.pt
[2023-08-02 00:41:19,604] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step30/mp_rank_00_model_states.pt...
[2023-08-02 00:41:19,646] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step30/mp_rank_00_model_states.pt.
[2023-08-02 00:41:19,647] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step30/zero_pp_rank_0_mp_rank_00_optim_states.pt...
[2023-08-02 00:41:19,853] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step30/zero_pp_rank_0_mp_rank_00_optim_states.pt.
[2023-08-02 00:41:19,853] [INFO] [engine.py:3285:_save_zero_checkpoint] zero checkpoint saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step30/zero_pp_rank_0_mp_rank_00_optim_states.pt
[2023-08-02 00:41:19,854] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step30 is ready now!
2023-08-02 00:41:19.854 | INFO     | __main__:log_dist:54 - [Rank 0] Saved model to experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg
[2023-08-02 00:41:20,525] [INFO] [logging.py:96:log_dist] [Rank 0] step=40, skipped=17, lr=[0.0001], mom=[(0.9, 0.999)]
[2023-08-02 00:41:20,526] [INFO] [timer.py:215:stop] epoch=0/micro_step=40/global_step=40, RunningAvgSamplesPerSec=165.2798057710019, CurrSamplesPerSec=141.28300869902063, MemAllocated=0.07GB, MaxMemAllocated=1.55GB
2023-08-02 00:41:20.526 | INFO     | __main__:log_dist:54 - [Rank 0] Loss: 10.6658
[2023-08-02 00:41:20,526] [INFO] [logging.py:96:log_dist] [Rank 0] [Torch] Checkpoint global_step40 is about to be saved!
[2023-08-02 00:41:20,529] [INFO] [logging.py:96:log_dist] [Rank 0] Saving model checkpoint: experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step40/mp_rank_00_model_states.pt
[2023-08-02 00:41:20,529] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step40/mp_rank_00_model_states.pt...
[2023-08-02 00:41:20,571] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step40/mp_rank_00_model_states.pt.
[2023-08-02 00:41:20,572] [INFO] [torch_checkpoint_engine.py:21:save] [Torch] Saving experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step40/zero_pp_rank_0_mp_rank_00_optim_states.pt...
[2023-08-02 00:41:20,778] [INFO] [torch_checkpoint_engine.py:23:save] [Torch] Saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step40/zero_pp_rank_0_mp_rank_00_optim_states.pt.
[2023-08-02 00:41:20,779] [INFO] [engine.py:3285:_save_zero_checkpoint] zero checkpoint saved experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg/global_step40/zero_pp_rank_0_mp_rank_00_optim_states.pt
[2023-08-02 00:41:20,779] [INFO] [torch_checkpoint_engine.py:33:commit] [Torch] Checkpoint global_step40 is ready now!
2023-08-02 00:41:20.779 | INFO     | __main__:log_dist:54 - [Rank 0] Saved model to experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg
experiments/bert_pretrain.2023.8.1.9.41.7.addjtvxg
(yk_py39) amd00@MZ32-00:~/yk_fork/DeepSpeedExamples/training/HelloDeepSpeed$ 
