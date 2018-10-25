// Based on: /home/riccardo/Documents/PhD/noise_variable_step_integrator/sim_cvode.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <sundials/sundials_types.h> /* definition of type realtype */
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <cvode/cvode_band.h>
#include <cvode/cvode_diag.h>
#include <sundials/sundials_math.h>

// #define DEBUG

/* Program constants */
#define true  1
#define false 0
#define NEQ   4              // number of equations
#define LMM   CV_ADAMS       // nonstiff
#define ITER  CV_FUNCTIONAL  // nonstiff
// #define LMM  CV_BDF         // stiff
// #define ITER CV_NEWTON      // stiff

#define ID_LINEAR 0
#define ID_DUFFING 1

/* Structure for user data */
typedef struct SimPara {
    /* Structure used to get the relevant simulation parameters from
    the Python code. The fields here MUST match the fields of the
    SimPara class in the Python code: same name, type and ORDER!
    */

    /* Circuit */
    realtype Cl;  // F
    realtype Cr;  // F
    realtype R1;  // ohm
    realtype L1;  // H
    realtype C1;  // F
    realtype R2;  // ohm
    realtype L2;  // H
    realtype C2;  // F

    /* Drive */
    int nr_drives;
    realtype* w_arr;  // rad/s
    realtype* A_arr;  // V
    realtype* P_arr;  // rad

    /* Select system */
    int sys_id;  // ID_LINEAR or ID_DUFFING

    /* Duffing: V/L * (1. - duff * V*V) */
    realtype duff;  // V^-2

    /* Thermal noise */
    int add_thermal_noise;  // bool
    realtype dt_noise;  // s
    realtype* noise1_array;  // V
    realtype* noise2_array;  // V
} SimPara;

/* Functions Called by the Solver */
// static realtype V_drive(realtype t, void* user_data);
static realtype V_dot_drive(realtype t, void* user_data);
// static realtype V_dotdot_drive(realtype t, void* user_data);
static realtype V_noise(realtype t, int node, void* user_data);
static int      ode_linear(realtype t, N_Vector y, N_Vector ydot, void* user_data);
static int      ode_duffing(realtype t, N_Vector y, N_Vector ydot, void* user_data);
static int      jac_linear(long int N, realtype t,
                        N_Vector y, N_Vector fy, DlsMat J, void* user_data,
                        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int      jac_duffing(long int N, realtype t,
                        N_Vector y, N_Vector fy, DlsMat J, void* user_data,
                        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);


/* Private function to check function return values */
static int check_flag(void *flagvalue, char *funcname, int opt);

/*
static realtype V_drive(realtype t, void* user_data) {
    SimPara* para = (SimPara*) user_data;
    int nr_drives = para->nr_drives;
    realtype* A_arr = para->A_arr;
    realtype* w_arr = para->w_arr;
    realtype* P_arr = para->P_arr;

    realtype ret = 0.;
    for (int i=0; i<nr_drives; i++) {
        ret += A_arr[i] * cos(w_arr[i]*t + P_arr[i]);
    }

    return(ret);
}
*/

static realtype V_dot_drive(realtype t, void* user_data) {
    SimPara* para = (SimPara*) user_data;
    int nr_drives = para->nr_drives;
    realtype* A_arr = para->A_arr;
    realtype* w_arr = para->w_arr;
    realtype* P_arr = para->P_arr;

    realtype ret = 0.;
    for (int i=0; i<nr_drives; i++) {
        ret += -w_arr[i] * A_arr[i] * sin(w_arr[i]*t + P_arr[i]);
    }

    return(ret);
}

/*
static realtype V_dotdot_drive(realtype t, void* user_data) {
    SimPara* para = (SimPara*) user_data;
    int nr_drives = para->nr_drives;
    realtype* A_arr = para->A_arr;
    realtype* w_arr = para->w_arr;
    realtype* P_arr = para->P_arr;

    realtype ret = 0.;
    for (int i=0; i<nr_drives; i++) {
        ret += -w_arr[i] * w_arr[i] * A_arr[i] * cos(w_arr[i]*t + P_arr[i]);
    }

    return(ret);
}
*/

static realtype V_noise(realtype t, int node, void* user_data) {
    SimPara* para = (SimPara*) user_data;
    realtype dt_noise = para->dt_noise;
    realtype* noise_array;
    if (node == 1){
        noise_array = para->noise1_array;
    } else {
        noise_array = para->noise2_array;
    }

    int idx = t / dt_noise;

    realtype dt = fmod(t, dt_noise);

    realtype dndt = (noise_array[idx + 1] - noise_array[idx]) / dt_noise;

    return(noise_array[idx] + dndt * dt);
}


static int ode_linear(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
    SimPara* para = (SimPara*) user_data;

    realtype P1 = NV_Ith_S(y, 0);
    realtype V1 = NV_Ith_S(y, 1);
    realtype P2 = NV_Ith_S(y, 2);
    realtype V2 = NV_Ith_S(y, 3);

    realtype V0_dot = V_dot_drive(t, para);
    realtype Vn1, Vn2;
    if (para->add_thermal_noise == true) {
        Vn1 = V_noise(t, 1, para);
        Vn2 = V_noise(t, 2, para);
    } else {
        Vn1 = 0.;
        Vn2 = 0.;
    }

    realtype Cl = para->Cl;
    realtype Cr = para->Cr;
    realtype R1 = para->R1;
    realtype L1 = para->L1;
    realtype C1 = para->C1;
    realtype R2 = para->R2;
    realtype L2 = para->L2;
    realtype C2 = para->C2;

    realtype Csum1 = C1 + Cl + Cr;
    realtype Csum2 = C2 + Cr;
    realtype g0 = 1. / (1. - Cr*Cr / (Csum1 * Csum2));
    realtype g1 = Cr / Csum1;
    realtype g2 = Cr / Csum2;

    NV_Ith_S(ydot, 0) = V1;
    NV_Ith_S(ydot, 1) = g0 * (-P1 / (Csum1 * L1) - (V1+Vn1) / (Csum1 * R1) + Cl / Csum1 * V0_dot + g1 * (-P2 / (Csum2 * L2) - (V2+Vn2) / (Csum2 * R2)));
    NV_Ith_S(ydot, 2) = V2;
    NV_Ith_S(ydot, 3) = g0 * (-P2 / (Csum2 * L2) - (V2+Vn2) / (Csum2 * R2) + g2 * (-P1 / (Csum1 * L1) - (V1+Vn1) / (Csum1 * R1) + Cl / Csum1 * V0_dot));

    return(0);
}


static int jac_linear(long int N, realtype t,
               N_Vector y, N_Vector fy, DlsMat J, void* user_data,
               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    SimPara* para = (SimPara*) user_data;

    realtype Cl = para->Cl;
    realtype Cr = para->Cr;
    realtype R1 = para->R1;
    realtype L1 = para->L1;
    realtype C1 = para->C1;
    realtype R2 = para->R2;
    realtype L2 = para->L2;
    realtype C2 = para->C2;

    realtype Csum1 = C1 + Cl + Cr;
    realtype Csum2 = C2 + Cr;
    realtype g0 = 1. / (1. - Cr*Cr / (Csum1 * Csum2));
    realtype g1 = Cr / Csum1;
    realtype g2 = Cr / Csum2;

    DENSE_ELEM(J, 0, 0) = RCONST(0.);
    DENSE_ELEM(J, 0, 1) = RCONST(1.);
    DENSE_ELEM(J, 0, 2) = RCONST(0.);
    DENSE_ELEM(J, 0, 3) = RCONST(0.);

    DENSE_ELEM(J, 1, 0) = -g0 / (Csum1 * L1);
    DENSE_ELEM(J, 1, 1) = -g0 / (Csum1 * R1);
    DENSE_ELEM(J, 1, 2) = -g0 * g1 / (Csum2 * L2);
    DENSE_ELEM(J, 1, 3) = -g0 * g1 / (Csum2 * R2);

    DENSE_ELEM(J, 2, 0) = RCONST(0.);
    DENSE_ELEM(J, 2, 1) = RCONST(0.);
    DENSE_ELEM(J, 2, 2) = RCONST(0.);
    DENSE_ELEM(J, 2, 3) = RCONST(1.);

    DENSE_ELEM(J, 3, 0) = -g0 * g2 / (Csum1 * L1);
    DENSE_ELEM(J, 3, 1) = -g0 * g2 / (Csum1 * R1);
    DENSE_ELEM(J, 3, 2) = -g0 / (Csum2 * L2);
    DENSE_ELEM(J, 3, 3) = -g0 / (Csum2 * R2);

    return(0);
}

static int ode_duffing(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
    SimPara* para = (SimPara*) user_data;

    realtype P1 = NV_Ith_S(y, 0);
    realtype V1 = NV_Ith_S(y, 1);
    realtype P2 = NV_Ith_S(y, 2);
    realtype V2 = NV_Ith_S(y, 3);

    realtype V0_dot = V_dot_drive(t, para);
    realtype Vn1, Vn2;
    if (para->add_thermal_noise == true) {
        Vn1 = V_noise(t, 1, para);
        Vn2 = V_noise(t, 2, para);
    } else {
        Vn1 = 0.;
        Vn2 = 0.;
    }

    realtype Cl = para->Cl;
    realtype Cr = para->Cr;
    realtype R1 = para->R1;
    realtype L1 = para->L1;
    realtype C1 = para->C1;
    realtype R2 = para->R2;
    realtype L2 = para->L2;
    realtype C2 = para->C2;
    realtype duff = para->duff;

    realtype Csum1 = C1 + Cl + Cr;
    realtype Csum2 = C2 + Cr;
    realtype g0 = 1. / (1. - Cr*Cr / (Csum1 * Csum2));
    realtype g1 = Cr / Csum1;
    realtype g2 = Cr / Csum2;

    NV_Ith_S(ydot, 0) = V1;
    NV_Ith_S(ydot, 1) = g0 * (-P1 / (Csum1 * L1) - (V1+Vn1) / (Csum1 * R1) + Cl / Csum1 * V0_dot + g1 * (-P2/(Csum2*L2)*(1.-duff*P2*P2) - (V2+Vn2) / (Csum2 * R2)));
    NV_Ith_S(ydot, 2) = V2;
    NV_Ith_S(ydot, 3) = g0 * (-P2/(Csum2*L2)*(1.-duff*P2*P2) - (V2+Vn2) / (Csum2 * R2) + g2 * (-P1 / (Csum1 * L1) - (V1+Vn1) / (Csum1 * R1) + Cl / Csum1 * V0_dot));

    return(0);
}

static int jac_duffing(long int N, realtype t,
               N_Vector y, N_Vector fy, DlsMat J, void* user_data,
               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    SimPara* para = (SimPara*) user_data;

    realtype P2 = NV_Ith_S(y, 2);

    realtype Cl = para->Cl;
    realtype Cr = para->Cr;
    realtype R1 = para->R1;
    realtype L1 = para->L1;
    realtype C1 = para->C1;
    realtype R2 = para->R2;
    realtype L2 = para->L2;
    realtype C2 = para->C2;
    realtype duff = para->duff;

    realtype Csum1 = C1 + Cl + Cr;
    realtype Csum2 = C2 + Cr;
    realtype g0 = 1. / (1. - Cr*Cr / (Csum1 * Csum2));
    realtype g1 = Cr / Csum1;
    realtype g2 = Cr / Csum2;

    DENSE_ELEM(J, 0, 0) = RCONST(0.);
    DENSE_ELEM(J, 0, 1) = RCONST(1.);
    DENSE_ELEM(J, 0, 2) = RCONST(0.);
    DENSE_ELEM(J, 0, 3) = RCONST(0.);

    DENSE_ELEM(J, 1, 0) = -g0 / (Csum1 * L1);
    DENSE_ELEM(J, 1, 1) = -g0 / (Csum1 * R1);
    DENSE_ELEM(J, 1, 2) = -g0 * g1 / (Csum2 * L2) * (1. - 2.*duff*P2);
    DENSE_ELEM(J, 1, 3) = -g0 * g1 / (Csum2 * R2);

    DENSE_ELEM(J, 2, 0) = RCONST(0.);
    DENSE_ELEM(J, 2, 1) = RCONST(0.);
    DENSE_ELEM(J, 2, 2) = RCONST(0.);
    DENSE_ELEM(J, 2, 3) = RCONST(1.);

    DENSE_ELEM(J, 3, 0) = -g0 * g2 / (Csum1 * L1);
    DENSE_ELEM(J, 3, 1) = -g0 * g2 / (Csum1 * R1);
    DENSE_ELEM(J, 3, 2) = -g0 / (Csum2 * L2) * (1. - 2.*duff*P2);
    DENSE_ELEM(J, 3, 3) = -g0 / (Csum2 * R2);

    return(0);
}


int integrate_cvode(void* user_data, realtype T0, realtype* y0,
                    realtype reltol, realtype abstol,
                    int nout, realtype dt, realtype* outdata) {
    int flag;
    N_Vector y;
    void* cvode_mem;

    cvode_mem = NULL;
    
    /* Create serial vector of length NEQ for I.C. */
    y = N_VNew_Serial(NEQ);
    if (check_flag((void *)y, "N_VNew_Serial", 0)) return(1);

    /* Initialize y */
    NV_Ith_S(y, 0) = y0[0];
    NV_Ith_S(y, 1) = y0[1];
    NV_Ith_S(y, 2) = y0[2];
    NV_Ith_S(y, 3) = y0[3];

    /* Call CVodeCreate to create the solver memory
     * with stepping and iteration method */
    cvode_mem = CVodeCreate(LMM, ITER);
    if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

    /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in y'=f(t,y), the inital time T0, and
    * the initial dependent variable vector y. */
    SimPara* para = (SimPara*) user_data;
    if (para->sys_id == ID_LINEAR) {
        flag = CVodeInit(cvode_mem, ode_linear, T0, y);
    } else if (para->sys_id == ID_DUFFING) {
        flag = CVodeInit(cvode_mem, ode_duffing, T0, y);
    } else {
        printf("*** UNKNOWN FORCE ID: %d ***\n", para->sys_id);
        return(1);
    }
    if (check_flag(&flag, "CVodeInit", 1)) return(1);

    /* Call CVodeSVtolerances to specify the scalar relative
     * and absolute tolerances */
    flag = CVodeSStolerances(cvode_mem, reltol, abstol);
    if (check_flag(&flag, "CVodeSStolerances", 1)) return(1);

    // /* Call CVodeRootInit to specify the root function g with 2 components */
    // flag = CVodeRootInit(cvode_mem, 2, g);
    // if (check_flag(&flag, "CVodeRootInit", 1)) return(1);

    /* Call CVDense to specify the CVDENSE dense linear solver */
    flag = CVDense(cvode_mem, NEQ);
    if (check_flag(&flag, "CVDense", 1)) return(1);

    /* Set the Jacobian routine (user-supplied) */
    if (para->sys_id == ID_LINEAR) {
        flag = CVDlsSetDenseJacFn(cvode_mem, jac_linear);
    } else if (para->sys_id == ID_DUFFING) {
        flag = CVDlsSetDenseJacFn(cvode_mem, jac_duffing);
    } else {
        printf("*** UNKNOWN FORCE ID: %d ***\n", para->sys_id);
        return(1);
    }
    if (check_flag(&flag, "CVDlsSetDenseJacFn", 1)) return(1);

    // Set optional inputs
    flag = CVodeSetUserData(cvode_mem, user_data);
    if(check_flag(&flag, "CVodeSetUserData", 1)) return(1);

    /* In loop, call CVode, write results, and test for error.
     Break out of loop when NOUT preset output times have been reached.  */
    int ii;
    realtype tout, tret;
    tout = T0 + dt;
    for(ii = 0; ii < nout; ii++){
        // Save result to output vectors
        // OBS: the 1st will be the given initial condition,
        // as in scipy.integrate.odeint
        outdata[ii*NEQ] = NV_Ith_S(y,0);     // P1, flux on node 1
        outdata[ii*NEQ + 1] = NV_Ith_S(y,1); // V1, voltage on node 1
        outdata[ii*NEQ + 2] = NV_Ith_S(y,2); // P2, flux on node 2
        outdata[ii*NEQ + 3] = NV_Ith_S(y,3); // V2, voltage on node 2

        flag = CVode(cvode_mem, tout, y, &tret, CV_NORMAL);

        // Treat roots
        while (flag==CV_ROOT_RETURN) {
            /*** do something here if wanted ***/
            // e.g. you could save the roots and return them...

            // Do time step
            flag = CVode(cvode_mem, tout, y, &tret, CV_NORMAL);
        }

        if (check_flag(&flag, "CVode", 1)) break;


        if (flag == CV_SUCCESS) {
            tout += dt;
        } else {
            // This shouldn't happen!
            // We already take care of roots
            // and there is check_flag, but you never know...
            printf("*** Something went wrong! ***\n");
            printf("Last error flag: %d \n", flag);
            break;
        }
    }

    // Update initial condition for next run
    // OBS: overwrites the user-provided y0!!
    y0[0] = NV_Ith_S(y,0); // P1, flux on node 1
    y0[1] = NV_Ith_S(y,1); // V1, voltage on node 1
    y0[2] = NV_Ith_S(y,2); // P2, flux on node 2
    y0[3] = NV_Ith_S(y,3); // V2, voltage on node 2

    /* Free y vector */
    N_VDestroy_Serial(y);

    /* Free integrator memory */
    CVodeFree(&cvode_mem);

    return(0);
}


/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns a flag so check if
 *            flag >= 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer 
 */

static int check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
        funcname);
    return(1); }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
          funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
        funcname);
    return(1); }

  return(0);
}
