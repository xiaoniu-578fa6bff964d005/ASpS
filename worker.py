import ray

import torch
import numpy as np
import transformers
import functools
import time

import lib


class MaxLengthLogitsProcessor(transformers.LogitsProcessor):
    def __init__(self, max_length, eos_token_id):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] > self.max_length:
            scores = torch.full_like(scores, -float("inf"))
            scores[:, self.eos_token_id] = 0.0
        return scores


class DummyWorker:
    def __init__(self, param):
        self.param = param

    @ray.method(concurrency_group="model")
    def __call__(self, d):
        return d


class Worker:
    def __init__(self, param):
        """
        param: {
            "model_str": str,
            "ref_model_str": str,
            "title": str,
            "device": str, # "cuda:0", "cpu", ...
        }
        """
        self.param = param
        load_model_kwargs = {
            "device_map": str(self.param["device"]),
            "pretrained_model_name_or_path": self.param["model_str"],
            "low_cpu_mem_usage": True,
        }
        if self.param["device"].startswith("cuda"):
            load_model_kwargs["torch_dtype"] = torch.float16
        load_ref_model_kwargs = {
            **load_model_kwargs,
            "pretrained_model_name_or_path": self.param["ref_model_str"],
        }
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            **load_model_kwargs
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.param["model_str"]
        )
        self.ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            **load_ref_model_kwargs
        )
        self.logits_processor = MaxLengthLogitsProcessor(1, self.tokenizer.eos_token_id)

    def process(self, d):
        """
        d: {
            "prompt": str,
            "seed": int,
            "method": str, # "basic", "mc", "tmc"
            "n": int,
            "max_length": int,
        }
        return: {
            # all input fields in d, except prompt
            "output_ids": list[int],
            "gen_seq_lens": list[int],
            "output": str,
            # timestamps
            "t_got_input": float,
            "t_got_first_output": float,
            "t_got_last_output": float,
        }
        """
        torch.manual_seed(d["seed"])
        self.logits_processor.max_length = d["max_length"]

        input_ids = self.tokenizer(d["prompt"], return_tensors="pt")["input_ids"].to(
            self.param["device"]
        )
        if d["method"] == "basic":
            generator = lib.basic_sample_generator
        elif d["method"] == "mc":
            generator = lib.mc_sample_generator
        elif d["method"] == "tmc":
            generator = lib.tmc_sample_generator
        else:
            raise ValueError(f"unknown sampling method {d['method']}")
        if d["method"] != "basic":
            generator = functools.partial(generator, ref_model=self.ref_model)

        gen = generator(
            model=self.model,
            input_ids=input_ids,
            n=d["n"],
            process_logits_kwargs={"logits_processor": self.logits_processor},
        )
        output_ids = []
        gen_seq_lens = []
        t_got_input = time.time()
        t_got_first_output = None
        for step_output_ids, step_output_logprobs in gen:
            if t_got_first_output is None:
                t_got_first_output = time.time()
            output_ids.extend(step_output_ids[0].cpu().tolist())
            gen_seq_lens.append(step_output_ids.shape[-1])
        t_got_last_output = time.time()
        output = self.tokenizer.decode(output_ids)

        return {
            **{k: v for k, v in d.items() if k != "prompt"},
            #  "output_ids": output_ids,
            "gen_seq_lens": gen_seq_lens,
            #  "output": output,
            "t_got_input": t_got_input,
            "t_got_first_output": t_got_first_output,
            "t_got_last_output": t_got_last_output,
        }

    @ray.method(concurrency_group="model")
    def __call__(self, d):
        """
        only batch_size=1 is supported
        """
        o = self.process({k: v[0] for k, v in d.items()})
        #  print("output", o)
        return {k: [v] for k, v in o.items()}
