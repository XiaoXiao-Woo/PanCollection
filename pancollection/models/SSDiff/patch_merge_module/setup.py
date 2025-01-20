# from setuptools import setup, Extension
# from Cython.Build import cythonize
# import sys

# python_version = f"{sys.version_info.major}{sys.version_info.minor}"
# print(f"Python version: {python_version}")


# module_name = f"PatchMergeModule_py{python_version}"

# extensions = [
#     Extension(module_name, ["PatchMergeModule.py"]),
# ]

# setup(
#     name=module_name,
#     ext_modules=cythonize(extensions),
# )

from setuptools import setup
from Cython.Build import cythonize
from pathlib import Path
import sys

# compile the cython code
# use: `python setup.py build_ext --inplace`
py_version = sys.version_info
output_dir = Path(__file__).parent / f"py{py_version.major}{py_version.minor}_module"
output_dir.mkdir(exist_ok=True)
setup(
    ext_modules=cythonize("scripts/tmp.py", 
                          language_level="3", 
                          compiler_directives={'linetrace': False, 'annotation_typing': False, 'emit_code_comments': False},
                          build_dir=output_dir.as_posix()),
    script_args=["build_ext", "--inplace"],
    package_dir={'model/module/patch_merge_module': f"py{py_version.major}{py_version.minor}_module"},
    name='patch_merge_module',
)

