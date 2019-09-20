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

all: cvode_transformer cvode_notch cvode_reflection cvode_transmission


clean: clean_transformer clean_notch clean_reflection clean_transmission


cvode_transformer: cvode_transformer.o
	$(CC) $(LDFLAGS) -o cvode_transformer.$(PYEXT) cvode_transformer.o $(LIBS)

cvode_transformer.o: cvode_transformer.c
	$(CC) $(CFLAGS) -c cvode_transformer.c

clean_transformer:
	rm cvode_transformer.o
	rm cvode_transformer.$(PYEXT)

cvode_notch: cvode_notch.o
	$(CC) $(LDFLAGS) -o cvode_notch.$(PYEXT) cvode_notch.o $(LIBS)

cvode_notch.o: cvode_notch.c
	$(CC) $(CFLAGS) -c cvode_notch.c

clean_notch:
	rm cvode_notch.o
	rm cvode_notch.$(PYEXT)

cvode_reflection: cvode_reflection.o
	$(CC) $(LDFLAGS) -o cvode_reflection.$(PYEXT) cvode_reflection.o $(LIBS)

cvode_reflection.o: cvode_reflection.c
	$(CC) $(CFLAGS) -c cvode_reflection.c

clean_reflection:
	rm cvode_reflection.o
	rm cvode_reflection.$(PYEXT)

cvode_transmission: cvode_transmission.o
	$(CC) $(LDFLAGS) -o cvode_transmission.$(PYEXT) cvode_transmission.o $(LIBS)

cvode_transmission.o: cvode_transmission.c
	$(CC) $(CFLAGS) -c cvode_transmission.c

clean_transmission:
	rm cvode_transmission.o
	rm cvode_transmission.$(PYEXT)
