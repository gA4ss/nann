#!/usr/bin/env python

from distutils.core import setup, Extension

# ext_modules=[Extension('Extest', sources=['testpy.c'])]

# nann c++ source codes
nann_sources = ['src/nanai_ann_nannmgr.cc',
                'src/nanai_ann_nanncalc.cc',
                'src/nanai_object.cc',
                'src/nanai_ann_nnn.cc',
                'src/nanai_memory.cc',
                'src/nanai_support.cc',
                'src/nanai_ann_alg_logistic.cc',
                'src/cJSON.cc',
                'src/pynann.cc',
                'src/nanmath/nanmath_vector.cc',
                'src/nanmath/nanmath_matrix.cc']

# nann h files
nann_h_files = ['inc','inc/nanmath']

# nann define macros
nann_define_macros = [('NDEBUG','1')]

# nann undefine macros
nann_undef_macros = []

# nann library dir
nann_library_dirs = []

# nann libraries
nann_libraries = []

# nann extera objects
nann_extra_objects = []

# nann extra compile args
nann_extra_compile_args = ['-std=c++11']

# nann extra link args
nann_extra_link_args = []

# nann extension
nann_extension = Extension('nann',
                           nann_sources,
                           include_dirs=nann_h_files,
                           define_macros=nann_define_macros,
                           undef_macros=nann_undef_macros,
                           library_dirs=nann_library_dirs,
                           libraries=nann_libraries,
                           extra_compile_args=nann_extra_compile_args,
                           extra_link_args=nann_extra_link_args)

# nann python interface source codes
nann_python_sources = ['']

setup(name='nann',
      version='1.0.0',
      description='Nanan rtificial neural network algorithm library',
      author='devilogic',
      author_email='logic.yan@me.com',
      url='https://git.coding.net/devilogic/nann.git',
      ext_modules=[nann_extension],
      packages=['pynann']
     )
