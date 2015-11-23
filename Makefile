
#
# Makefile - needs GNU make 3.81 or better
#
# Copyright (C) 2013-2014 4dog.cn
#

# 确定是否是正确的Make版本
ifneq ($(findstring $(firstword $(MAKE_VERSION)),3.77 3.78 3.78.1 3.79 3.79.1 3.80),)
$(error GNU make 3.81 or better is required)
endif

# 定义源目录,取最后一个make文件的,也就是当前处理的make文件的路径作为源目录
# sed的作用是如果文件名以$结尾,则去掉这个$号
ifndef srcdir
srcdir := $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST)))
srcdir := $(shell echo '$(srcdir)' | sed 's,/*$$,,')
endif

# 设定顶级目录
ifndef top_srcdir
top_srcdir := $(srcdir)/..
endif

# 如果源目录非当前目录则设定VPATH
ifneq ($(srcdir),.)
##$(info Info: using VPATH . $(srcdir))
VPATH := . $(srcdir)
endif

# 包含全局配置与本地配置脚本
# include $(wildcard $(top_srcdir)/Makevars.global $(srcdir)/Makevars.local)
# include $(srcdir)/Makevars.local

# 全局配置选项
PLATFORM := $(shell uname | sed -e 's/_.*//')

#
# 一些优先工程配置的宏设定
#

# 是否开启调试模式
#DEBUG ?= 1

# 是否使用GNU的编译器
#USE_GUNC ?= 1

# 平台是否是32位
ARCH ?= 32

# 指定是Android平台编译
#ANDROID ?= 1

# ARM:1,X86:2,MIPS:3
CPU ?= 1

# 设置工程名称
project ?= tdog_cipher

#
# 工具设置
#

ifndef COPY
COPY=cp
endif

ifndef RM
RM=rm
endif

ifndef MAKE
MAKE=make
endif


# 判断编译器集合
ifeq ($(ANDROID),1)

# 设定NDK
NDK_HOME ?= ~/sdk/android-ndk-r8e

ifeq ($(CPU),1)
# ARM编译
SYSROOT := $(NDK_HOME)/platforms/android-8/arch-arm/
ADDR2LINE := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-addr2line
AR := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-ar
AS := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-as
CC := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-gcc --sysroot=$(SYSROOT)
CXX := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-g++ --sysroot=$(SYSROOT)
GDB := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-gdb
GDBTUI := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-gdbtui
GPROF := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-gprof
LD := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-ld --sysroot=$(SYSROOT)
NM := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-nm
OBJCOPY := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-objcopy
OBJDUMP := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-objdump
RANLIB := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-ranlib
READELF := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-readelf
RUN := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-run
SIZE := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-size
STRINGS := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-strings
STRIP := $(NDK_HOME)/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86_64/bin/arm-linux-androideabi-strip
else
# X86编译
SYSROOT := $(NDK_HOME)/platforms/android-14/arch-x86g/
ADDR2LINE := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-addr2line
AR := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-ar
AS := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-as
CC := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-gcc --sysroot=$(SYSROOT)
CXX := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-g++ --sysroot=$(SYSROOT)
GDB := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-gdb
GDBTUI := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-gdbtui
GPROF := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-gprof
LD := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-ld --sysroot=$(SYSROOT)
NM := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-nm
OBJCOPY := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-objcopy
OBJDUMP := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-objdump
RANLIB := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-ranlib
READELF := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-readelf
RUN := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-run
SIZE := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-size
STRINGS := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-strings
STRIP := $(NDK_HOME)/toolchains/x86-4.4.3/prebuilt/linux-x86_64/bin/i686-linux-android-strip
endif
else
ADDR2LINE := addr2line
AR := ar
AS := as
CC := gcc
CXX := g++
GDB := gdb
GDBTUI :=
GPROF :=  gprof
LD := ld
NM := nm
OBJCOPY := objcopy
OBJDUMP := objdump
RANLIB := ranlib
READELF := readelf
RUN := run
SIZE := size
STRINGS := strings
STRIP := strip
endif

# 交叉工具
objdump: $(OBJDUMP)
	$(OBJDUMP) $(CMD)

objcopy: $(OBJCOPY)
	$(OBJCOPY) $(CMD)

readelf: $(READELF)
	$(READELF) $(CMD)

nm: $(NM)
	$(NM) $(CMD)

# 使用GNUC编译器
#ifneq ($(findstring $(firstword $(CXX)),g++),)
USE_GNUC ?= 1
#endif

# 如果USE_GNUC等于1则设定相应编译选项
ifeq ($(USE_GNUC),1)

# 调试编译
ifeq ($(DEBUG),1)
DEFS += -DCDOG_DEBUG=1
else
DEFS += -DCDOG_DEBUG=0 -DNODEBUG
endif

# 一些自定义的宏设定
DEFS += -DUSE_GNUC=1

# 其余错误编译选项
# CXXFLAGS_WERROR = -Werror
# -Wcast-qual 加上这个选项,类型转换很受限制
# -Wcast-align 类型转换对齐粒度受限制
CXXFLAGS += -Wall -Wpointer-arith -Wshadow -Wwrite-strings -W 
CXXFLAGS += -Wno-unused-function
CXXFLAGS += $(CXXFLAGS_WERROR)

# 编译选项
CXXFLAGS += -fpie

# 打包选项
ARFLAGS += rcs

else

#
# 不使用GNU的编译器
#

endif

#
# 一些全局的库与代码
#

ifeq ($(ANDROID),1)
endif

global_OBJECTS += 
global_SOURCES +=

# -r(--no-builtin-rules)禁止make使用任何隐含规则
# -R(--no-builtin-variabes)禁止make使用任何作用于变量上的隐含规则
MAKEFLAGS += -rR
.SUFFIXES:
export SHELL = /bin/sh

# call函数的参数模板e
# $($1)负责展开选项
# $(EXTRA_$1)表示当前选项的扩展选项,例如:CXXFLAGS,则展开变量$(EXTRA_CXXFLAGS)
# $($(project)_$1)表示针对tdog的选项,例如:CXXFLAGS,则展开变量$(tdog_CXXFLAGS)
# $($(basename $(notdir $@)).$1) 表示针对某个文件的选项,例如:CXXFLAGS,$@=linker.cpp
# 则展开变量$(linker.CXXFLAGS)
# 针对几个层级进行编译或者链接或者其他操作的参数构造
override e = $($1) $(EXTRA_$1) $($(project)_$1) $($(basename $(notdir $@)).$1)

# 指定编译器
# ifeq ($(CXX),)
# CXX = g++
# endif

# 如果USE_GNUC之前定义过,则保持之前的值,否则则默认开启
# ifneq ($(findstring $(firstword $(CXX)),g++),)
# USE_GNUC ?= 1
# endif

# 如果USE_GNUC等于1则设定相应编译选项
ifeq ($(USE_GNUC),1)

# 调试编译
ifeq ($(DEBUG),1)
CXXFLAGS += -O0 -g3
else
CXXFLAGS += -O3
endif

# 体系架构
ifeq ($(ANDROID),)
# 在真实的机器下采用指定体系的编译选项
ifeq ($(ARCH),32)
CXXFLAGS += -m32
else
CXXFLAGS += -m64
endif
endif

endif

# 合成编译选项
CPPFLAGS += $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) # 生成依赖关系使用
CXXLD ?= $(CXX)

# 后缀选项
exeext ?= .so
libext ?= .a
objext ?= .o

# 默认后缀
defext ?= $(exeext)

# 文件集合
local_SOURCES := $(sort $(wildcard $(srcdir)/*.cc))
local_OBJECTS := $(notdir $(local_SOURCES:.cc=$(objext)))

# 目标
so: $(project)$(defext) | .depend
.DELETE_ON_ERROR: $(project)$(defext) $(local_OBJECTS) .depend   # 发生错误时删除

lib: lib$(project)$(libext) | .depend
.DELETE_ON_ERROR: lib$(project)$(libext) $(local_OBJECTS) .depend   # 发生错误时删除

# 这里提供了外部控制的HOOK选项
# 通过project.out变量的值.PRE_LINK_STEP来进行控制
# 当链接完成后可由project.out变量.POST_LINK_STEP来进行控制
$(project)$(defext): $(local_OBJECTS) $($(project)_DEPENDENCIES)
	$($(notdir $@).PRE_LINK_STEP)
	$(strip $(CXXLD) $(call e,CPPFLAGS) $(call e,CXXFLAGS) $(call e,LDFLAGS) -shared -o $@ $(local_OBJECTS) $(global_OBJECTS) $(call e,LDADD) $(call e,LIBS))
	$($(notdir $@).POST_LINK_STEP)

lib$(project)$(libext): $(local_OBJECTS) $($(project)_DEPENDENCIES)
	$($(notdir $@).PRE_LINK_STEP)
	$(strip $(AR) $(ARFLAGS) $@ $(local_OBJECTS) $(global_OBJECTS))
	$($(notdir $@).POST_LINK_STEP)

%.o : %.cc | .depend
	$(strip $(CXX) $(call e,CPPFLAGS) $(call e,CXXFLAGS) -o $@ -c $<)

# 生成依赖文件
.depend: $(sort $(wildcard $(srcdir)/*.cc $(srcdir)/*.h)) $(MAKEFILE_LIST)

# 如果是GNU编译器集合
# 从文件集合中取出.cc文件依次进行编译,并将编译输出去掉首尾空格写入到.depend文件中
# 如果非GNU编译器单纯的创建一个.depend的文件
ifeq ($(USE_GNUC),1)
	@echo "Updating $@"
	@$(strip $(CXX) $(call e,CPPFLAGS) -MM) $(filter %.cc,$^) > $@
else
	touch $@
endif

# 编译文档
ifneq ($(BUILD_DOC),)
	$(MAKE) -C doc $@
endif

# 清除
mostlyclean clean distclean maintainer-clean:
	rm -f *.d *.map *.o *.obj *.res .depend $(project)$(defext) $(project)$(libext) $(project).ttp $(project)$(exeext) _version.h

# 伪目标
.PHONY: default all mostlyclean clean distclean maintainer-clean

ifeq ($(MAKECMDGOALS),mostlyclean)
else ifeq ($(MAKECMDGOALS),clean)
else ifeq ($(MAKECMDGOALS),distclean)
else ifeq ($(MAKECMDGOALS),maintainer-clean)
else
-include .depend
endif
