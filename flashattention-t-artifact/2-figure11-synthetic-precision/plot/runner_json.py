from dataclasses import dataclass
from enum import Enum
import json

class SDPAVariant(Enum):
    FP16_MATH = "fp16_math"
    FP64_MATH = "fp64_math"
    FP16_FA = "fp16_fa"
    FP16_FULLSIMT = "fp16_fullsimt"
    FP16_FULLMMA = "fp16_fullmma"
    FP16_ILPH = "fp16_ilph"
    FP16_ILPV = "fp16_ilpv"

@dataclass
class SDPACompareResultEntry:
    ref_method: SDPAVariant
    target_method: SDPAVariant
    mse: float
    rmse: float

    def to_json(self):
        return {
            "ref_method": self.ref_method.value,
            "target_method": self.target_method.value,
            "mse": self.mse,
            "rmse": self.rmse
        }
    
    @classmethod
    def from_json(cls, json_data):
        return cls(
            ref_method=SDPAVariant(json_data["ref_method"]),
            target_method=SDPAVariant(json_data["target_method"]),
            mse=json_data["mse"],
            rmse=json_data["rmse"]
        )

@dataclass
class PrecisionTestResult:
    tensorgen_dtype: str
    batchsize: int
    seqlen: int
    nheads: int
    headdim: int
    base_variance: float
    outlier_variance: float
    seed: int
    compare_results: list[SDPACompareResultEntry]

    def to_json(self):
        return {
            "tensorgen_dtype": self.tensorgen_dtype,
            "batchsize": self.batchsize,
            "seqlen": self.seqlen,
            "nheads": self.nheads,
            "headdim": self.headdim,
            "base_variance": self.base_variance,
            "outlier_variance": self.outlier_variance,
            "seed": self.seed,
            "compare_results": [result.to_json() for result in self.compare_results]
        }
    
    @classmethod
    def from_json(cls, json_data):
        return PrecisionTestResult(
            tensorgen_dtype=json_data["tensorgen_dtype"],
            batchsize=json_data["batchsize"],
            seqlen=json_data["seqlen"],
            nheads=json_data["nheads"],
            headdim=json_data["headdim"],
            base_variance=json_data["base_variance"],
            outlier_variance=json_data["outlier_variance"],
            seed=json_data["seed"],
            compare_results=[SDPACompareResultEntry.from_json(result) for result in json_data["compare_results"]]
        )
    
@dataclass
class LoopPrecisionTestResult:
    results: list[PrecisionTestResult]
    
    def to_json(self):
        return {
            "results": [result.to_json() for result in self.results]
        }
    
    @classmethod
    def from_json(cls, json_data):
        return LoopPrecisionTestResult(
            results=[PrecisionTestResult.from_json(result) for result in json_data["results"]]
        )
    
if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run directly. It is meant to be imported and used in other scripts.")