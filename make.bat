@echo off
if "%1"=="" GOTO COMPILE
if "%1"=="clean" GOTO CLEAN
GOTO HELP

:CLEAN
del sim_cvode.o
del sim_cvode.dll
GOTO DONE

:COMPILE

set INCLUDES=-I"win64/include"
set CFLAGS=-std=gnu99 -Wall
set LDFLAGS=-shared
set LIBS= -L"win64/lib" -lsundials_cvode -lsundials_nvecserial -lgsl -lm

echo on

cmd /C gcc %CFLAGS% %INCLUDES% -c sim_cvode.c -o sim_cvode.o
cmd /C gcc %CFLAGS% %LDFLAGS% -o sim_cvode.dll sim_cvode.o %LIBS%

@echo off
GOTO DONE

:HELP
echo Type 'make' to compile or 'make clean' to remove compiled files
GOTO DONE

:DONE
