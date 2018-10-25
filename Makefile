sim_cvode.so: sim_cvode.o
	gcc -Wall -shared -o sim_cvode.so sim_cvode.o -lsundials_cvode -lsundials_nvecserial

sim_cvode.o: sim_cvode.c
	gcc -Wall -fPIC -c sim_cvode.c

clean:
	rm sim_cvode.o
	rm sim_cvode.so
