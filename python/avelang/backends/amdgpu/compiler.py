from dataclasses import dataclass
import os

from ..compiler import BaseBackend, GPUTarget
from ...compiler.code_generator import compile_to_binary


@dataclass(frozen=True)
class AmdgpuCompilerOptions:
    num_warps: int = -1
    validate_invariants: bool = False


class AmdgpuCompiler(BaseBackend):
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.tuple == "amdgcn-amd-amdhsa"

    def parse_options(self, options) -> object:
        if options is None:
            return AmdgpuCompilerOptions()
        if isinstance(options, AmdgpuCompilerOptions):
            return options

        args = {
            "validate_invariants": os.environ.get("AVELANG_VALIDATE_INVARIANTS", "").lower()
            in ("1", "true", "yes", "on"),
        }
        if "num_warps" in options and options["num_warps"] is not None:
            args["num_warps"] = options["num_warps"]
        if "validate_invariants" in options and options["validate_invariants"] is not None:
            args["validate_invariants"] = bool(options["validate_invariants"])
        return AmdgpuCompilerOptions(**args)

    def compile(self, src, target, options=None):
        if options is None:
            options = self.parse_options({})
        elif isinstance(options, dict):
            options = self.parse_options(options)
        return compile_to_binary(src, target, opt_level=2, options=options)
