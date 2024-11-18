import ray

import os
import itertools

large_llamas = [
    "daryl149/llama-2-7b-chat-hf",
    "daryl149/llama-2-13b-chat-hf",
    "daryl149/llama-2-70b-chat-hf",
    "huggyllama/llama-7b",
    "huggyllama/llama-13b",
    "huggyllama/llama-65b",
]
small_llamas = ["JackFram/llama-68m", "JackFram/llama-160m"]

large_opts = [
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    "facebook/opt-66b",
]
small_opts = ["facebook/opt-125m", "facebook/opt-350m"]

large_gpts = [
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neo-20B",
]
small_gpts = ["gpt2", "gpt2-medium"]


def to_display_model_name(s):
    if "/" in s:
        s = s.split("/")[-1]
    s = s.replace("-chat-hf", "")
    return s


import tasks

from worker import Worker

#  from worker import DummyWorker as Worker

#  ns = [1, 2, 3, 4, 5, 7, 9]
ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
seeds = list(range(10))
#  ns = [2]
#  seeds = [0]
repartition_size = 500


def exp_translation_scan_n():
    ds = tasks.get_translation_ds()
    for model_str, ref_model_str in itertools.product(
        #  large_llamas[:1],
        #  large_llamas[:1] + large_llamas[3:4],
        #  large_llamas[3:4] + large_llamas[:1],
        large_llamas[3:4],
        small_llamas[:1],
    ):
        worker_param = {
            "model_str": model_str,
            "ref_model_str": ref_model_str,
            "task": "translation_scan_n",
            "device": "cuda:0",
        }
        save_path = os.path.join(
            os.path.dirname(__file__),
            "data_root",
            "data",
            "translation_scan_n",
            f"{to_display_model_name(model_str)}_{to_display_model_name(ref_model_str)}",
        )
        outds = (
            ray.data.from_huggingface(ds)
            .repartition(repartition_size)
            .map(lambda d: {**d, "max_length": 128})
            .flat_map(
                lambda d: [{**d, "method": "basic", "n": 1}]
                + [{**d, "method": "mc", "n": n} for n in ns]
                + [{**d, "method": "tmc", "n": n} for n in ns]
            )
            .flat_map(lambda d: [{**d, "seed": seed} for seed in seeds])
            .map_batches(
                Worker,
                batch_size=1,
                 compute=ray.data.ActorPoolStrategy(max_tasks_in_flight_per_actor=5),
                fn_constructor_kwargs={"param": worker_param},
                # remote_args
                num_gpus=1,
                max_restarts=0,
                concurrency_groups={"model": 1},
            )
        )

        outds.write_parquet(save_path)


def exp_summarization_scan_n():
    ds = tasks.get_summarization_ds()
    for model_str, ref_model_str in itertools.product(
        large_llamas[3:4],
        small_llamas[:1],
    ):
        worker_param = {
            "model_str": model_str,
            "ref_model_str": ref_model_str,
            "task": "summarization_scan_n",
            "device": "cuda:0",
        }
        save_path = os.path.join(
            os.path.dirname(__file__),
            "data_root",
            "data",
            "summarization_scan_n",
            f"{to_display_model_name(model_str)}_{to_display_model_name(ref_model_str)}",
        )
        outds = (
            ray.data.from_huggingface(ds)
            .repartition(repartition_size)
            .map(lambda d: {**d, "max_length": 128})
            .flat_map(
                lambda d: [{**d, "method": "basic", "n": 1}]
                + [{**d, "method": "mc", "n": n} for n in ns]
                + [{**d, "method": "tmc", "n": n} for n in ns]
            )
            .flat_map(lambda d: [{**d, "seed": seed} for seed in seeds])
            .map_batches(
                Worker,
                batch_size=1,
                compute=ray.data.ActorPoolStrategy(),
                fn_constructor_kwargs={"param": worker_param},
                # remote_args
                num_gpus=1,
                max_restarts=0,
                concurrency_groups={"model": 1},
            )
        )

        outds.write_parquet(save_path)


def exp_oeg_scan_n_opt():
    ds = tasks.get_oeg_ds()
    for model_str, ref_model_str in itertools.product(
        large_opts[:1],
        small_opts[:1],
    ):
        worker_param = {
            "model_str": model_str,
            "ref_model_str": ref_model_str,
            "task": "oeg_scan_n",
            "device": "cuda:0",
        }
        save_path = os.path.join(
            os.path.dirname(__file__),
            "data_root",
            "data",
            "oeg_scan_n",
            f"{to_display_model_name(model_str)}_{to_display_model_name(ref_model_str)}",
        )
        outds = (
            ray.data.from_huggingface(ds)
            .repartition(repartition_size)
            .map(lambda d: {**d, "max_length": 128})
            .flat_map(
                lambda d: [{**d, "method": "basic", "n": 1}]
                + [{**d, "method": "mc", "n": n} for n in ns]
                + [{**d, "method": "tmc", "n": n} for n in ns]
            )
            .flat_map(lambda d: [{**d, "seed": seed} for seed in seeds])
            .map_batches(
                Worker,
                batch_size=1,
                compute=ray.data.ActorPoolStrategy(),
                fn_constructor_kwargs={"param": worker_param},
                # remote_args
                num_gpus=1,
                max_restarts=0,
                concurrency_groups={"model": 1},
            )
        )

        outds.write_parquet(save_path)


def exp_oeg_scan_n_gptgpu():
    ds = tasks.get_oeg_ds()
    for model_str, ref_model_str in itertools.product(
        large_gpts[:1],
        small_gpts[:1],
    ):
        worker_param = {
            "model_str": model_str,
            "ref_model_str": ref_model_str,
            "task": "oeg_scan_n",
            "device": "cuda:0",
        }
        save_path = os.path.join(
            os.path.dirname(__file__),
            "data_root",
            "data",
            "oeg_scan_n",
            f"{to_display_model_name(model_str)}_{to_display_model_name(ref_model_str)}",
        )
        outds = (
            ray.data.from_huggingface(ds)
            .repartition(repartition_size)
            .map(lambda d: {**d, "max_length": 128})
            .flat_map(
                lambda d: [{**d, "method": "basic", "n": 1}]
                + [{**d, "method": "mc", "n": n} for n in ns]
                + [{**d, "method": "tmc", "n": n} for n in ns]
            )
            .flat_map(lambda d: [{**d, "seed": seed} for seed in seeds])
            .map_batches(
                Worker,
                batch_size=1,
                compute=ray.data.ActorPoolStrategy(),
                fn_constructor_kwargs={"param": worker_param},
                # remote_args
                num_gpus=1,
                max_restarts=0,
                concurrency_groups={"model": 1},
            )
        )

        outds.write_parquet(save_path)


def exp_oeg_scan_n_gptcpu():
    ds = tasks.get_oeg_ds()
    for model_str, ref_model_str in itertools.product(
        large_opts[:1],
        small_opts[:1],
    ):
        worker_param = {
            "model_str": model_str,
            "ref_model_str": ref_model_str,
            "task": "oeg_scan_n",
            "device": "cpu",
        }
        save_path = os.path.join(
            os.path.dirname(__file__),
            "data_root",
            "data",
            "oeg_scan_n",
            f"cpu_{to_display_model_name(model_str)}_{to_display_model_name(ref_model_str)}",
        )
        outds = (
            ray.data.from_huggingface(ds)
            .repartition(repartition_size)
            .map(lambda d: {**d, "max_length": 128})
            .flat_map(
                lambda d: [{**d, "method": "basic", "n": 1}]
                + [{**d, "method": "mc", "n": n} for n in ns]
                + [{**d, "method": "tmc", "n": n} for n in ns]
            )
            .flat_map(lambda d: [{**d, "seed": seed} for seed in seeds])
            .map_batches(
                Worker,
                batch_size=1,
                compute=ray.data.ActorPoolStrategy(),
                fn_constructor_kwargs={"param": worker_param},
                # remote_args
                num_cpus=2,
                max_restarts=0,
                concurrency_groups={"model": 1},
            )
        )

        outds.write_parquet(save_path)


if __name__ == "__main__":
    ray.init()

    ray.data.DatasetContext.get_current().execution_options.preserve_order = True
    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    # translation task scan n
    exp_translation_scan_n()
    # translation task fix n
    # summarization task scan n
    exp_summarization_scan_n()
    # open-ended generation task scan n, opt
    exp_oeg_scan_n_opt()
    # open-ended generation task, gpt
    exp_oeg_scan_n_gptgpu()
    #  exp_oeg_scan_n_gptcpu()

    ray.shutdown()
