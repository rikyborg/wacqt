# Figure out what os
ifeq "$(OS)" "Windows_NT"
    UNAME=Windows_NT
else
    UNAME=$(shell uname -s)
endif

$(info System is $(UNAME))

ifeq "$(UNAME)" "Linux"
    CC = gcc
    CFLAGS = -Wall -fPIC -O3
    LDFLAGS = -Wall -shared
    LIBS = -lsundials_cvode -lsundials_nvecserial -lgsl -lm
    PYEXT = so
    RM = rm
endif
ifeq "$(UNAME)" "Darwin"
    CC = clang
    CFLAGS = -Wall -fPIC -I"/usr/local/include"
    LDFLAGS = -bundle
    LIBS = -lsundials_cvode -lsundials_nvecserial -lgsl
    PYEXT = bundle
    RM = rm
endif

all: sim_cvode


sim_cvode: sim_cvode.o
	$(CC) $(LDFLAGS) -o sim_cvode.$(PYEXT) sim_cvode.o $(LIBS)

sim_cvode.o: sim_cvode.c
	$(CC) $(CFLAGS) -c sim_cvode.c

clean:
	rm sim_cvode.o
	rm sim_cvode.$(PYEXT)
