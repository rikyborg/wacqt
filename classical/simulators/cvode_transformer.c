// Based on: /home/riccardo/Documents/PhD/noise_variable_step_integrator/sim_cvode.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cvode/cvode.h>               /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <cvode/cvode_direct.h>        /* access to CVDls interface            */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype      */

#include <gsl/gsl_spline.h>

// #define DEBUG

/* Program constants */
#define true  1
#define false 0
#define NEQ   3              // number of equations

/*  // Use instead para->stiff_equation
#define LMM   CV_ADAMS       // nonstiff
#define ITER  CV_FUNCTIONAL  // nonstiff
// #define LMM  CV_BDF         // stiff
// #define ITER CV_NEWTON      // stiff
*/

#define ID_LINEAR 0
#define ID_DUFFING 1
#define ID_JOSEPHSON 2

#define ID_NO_DRIVE 0
#define ID_LOCKIN 1
#define ID_DRIVE_V 2

// #define PHI0 2.067833831e-15  // Wb, magnetic flux quantum
// #define PHI0 2.067833831e-12  // FAKE


/* Structure for user data */
typedef struct SimPara {
    /* Structure used to get the relevant simulation parameters from
    the Python code. The fields here MUST match the fields of the
    SimPara class in the Python code: same name, type and ORDER!
    */

    /* Solver */
    int stiff_equation;  // bool

    /* Circuit */
    realtype R0;  // ohm
    realtype L0;  // H
    realtype Mg;  // H
    realtype L1;  // H
    realtype C1;  // F
    realtype R1;  // ohm
    realtype R2;  // ohm

    /* Drive */
    int drive_id;  // ID_LOCKIN, ID_DRIVE_V
    // Lockin
    int nr_drives;
    realtype* w_arr;  // rad/s
    realtype* A_arr;  // V
    realtype* P_arr;  // rad
    // Drive_V or Drive_dVdt
    realtype* drive_V_arr;  // V or V/s
    gsl_spline* drive_spline;  // internal
    gsl_interp_accel* drive_acc;  // internal

    /* Select system */
    int sys_id;  // ID_LINEAR, ID_DUFFING, ID_JOSEPHSON or ID_JOSEPHSON_BOTH

    /* Duffing: V/L * (1. - duff * V*V) */
    realtype duff1;  // V^-2

    /* Josephson */
    realtype phi0;

    /* Thermal noise */
    int add_thermal_noise;  // bool
    realtype* noise0_arr;  // V
    gsl_spline* noise0_spline;  // internal
    gsl_interp_accel* noise0_acc;  // internal
    realtype* noise1_arr;  // V
    gsl_spline* noise1_spline;  // internal
    gsl_interp_accel* noise1_acc;  // internal
    realtype* noise2_arr;  // V
    gsl_spline* noise2_spline;  // internal
    gsl_interp_accel* noise2_acc;  // internal

    /* Other internal */
    realtype b[NEQ];
    realtype a[NEQ][NEQ];
} SimPara;


static int init_para(SimPara* para) {
    realtype r0 = para->R0;  // ohm
    realtype l0 = para->L0;  // H
    realtype l1 = para->L1;  // H
    realtype c1 = para->C1;  // F
    realtype r1 = para->R1;  // ohm
    realtype r2 = para->R2;  // ohm
    realtype mG = para->Mg;  // H

    /* printf("r0 = %f \n", r0); */
    /* printf("l0 = %f \n", l0); */
    /* printf("l1 = %f \n", l1); */
    /* printf("c1 = %f \n", c1); */
    /* printf("r1 = %f \n", r1); */
    /* printf("r2 = %f \n", r2); */
    /* printf("mG = %f \n", mG); */

    para->b[0] = l1 / (l0*l1 - mG*mG);
    para->b[1] = 0.;
    para->b[2] = 0.;

    para->a[0][0] = -l1*(r0 + r2) / (l0*l1 - mG*mG);
    para->a[0][1] = 0.;
    para->a[0][2] = -mG / (l0*l1 - mG*mG);

    para->a[1][0] = 0.;
    para->a[1][1] = 0.;
    para->a[1][2] = 1.;

    para->a[2][0] = mG / (c1*l1);
    para->a[2][1] = -1. / (c1*l1);
    para->a[2][2] = -1. / (c1*r1);

    /* for (int ii=0;ii<NEQ;ii++) { */
    /*   printf("b[%d] = %g\n", ii, para->b[ii]); */
    /* } */
    /* for (int ii=0;ii<NEQ;ii++) { */
    /*   printf("\n"); */
    /*   for (int jj=0;jj<NEQ;jj++){ */
    /*     printf("a[%d][%d] = %g\n", ii, jj, para->a[ii][jj]); */
    /*   } */
    /* } */

    return(0);
}

/* Functions Called by the Solver */
static realtype V_drive(realtype t, void* user_data);
static realtype V_noise(realtype t, int node, void* user_data);


/* Private function to check function return values */
static int check_flag(void *flagvalue, char *funcname, int opt);

static realtype V_drive(realtype t, void* user_data) {
    SimPara* para = (SimPara*) user_data;
    int drive_id = para->drive_id;
    if (drive_id==ID_NO_DRIVE) {
        return 0.;
    } else if (drive_id==ID_LOCKIN) {
        int nr_drives = para->nr_drives;
        realtype* A_arr = para->A_arr;
        realtype* w_arr = para->w_arr;
        realtype* P_arr = para->P_arr;

        realtype ret = 0.;
        for (int i=0; i<nr_drives; i++) {
            ret += A_arr[i] * cos(w_arr[i]*t + P_arr[i]);
        }

        return ret;
    } else { // drive_id==ID_DRIVE_V
        return gsl_spline_eval(para->drive_spline, t, para->drive_acc);
    }
}

static realtype V_noise(realtype t, int node, void* user_data) {
    SimPara* para = (SimPara*) user_data;
    if (node == 0) {
      return gsl_spline_eval(para->noise0_spline, t, para->noise0_acc);
    } else if (node == 1) {
      return gsl_spline_eval(para->noise1_spline, t, para->noise1_acc);
    } else if (node == 2) {
      return gsl_spline_eval(para->noise2_spline, t, para->noise2_acc);
    } else {
      printf("\n\n\n*** Error in V_noise!!!\n\n\n");
      return 0.;
    }
}


static int ode_linear(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
    SimPara* para = (SimPara*) user_data;

    realtype Vg = V_drive(t, para);

    for (int ii=0; ii<NEQ; ii++) {
        NV_Ith_S(ydot, ii) = 0.;
        for (int jj=0; jj<NEQ; jj++) {
            NV_Ith_S(ydot, ii) += para->a[ii][jj] * NV_Ith_S(y, jj);
        }
        NV_Ith_S(ydot, ii) += para->b[ii] * Vg;
    }

    /* Add noise */
    if (para->add_thermal_noise == true) {
        realtype Vn0 = V_noise(t, 0, para);
        realtype Vn1 = V_noise(t, 1, para);
        realtype Vn2 = V_noise(t, 2, para);

        NV_Ith_S(ydot, 0) += para->b[0] * Vn0;
        NV_Ith_S(ydot, 0) += -para->b[0] * Vn2;
        NV_Ith_S(ydot, 2) += -para->a[2][2] * Vn1;
    }

    return(0);
}


static int jac_linear(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                      void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    SimPara* para = (SimPara*) user_data;

    for (int ii=0; ii<NEQ; ii++) {
        for (int jj=0; jj<NEQ; jj++) {
            SM_ELEMENT_D(J, ii, jj) = para->a[ii][jj];
        }
    }

    return(0);
}

static int ode_duffing(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
    SimPara* para = (SimPara*) user_data;

    /* First do the linear part */
    ode_linear(t, y, ydot, user_data);

    /* Then add the Duffing terms */
    realtype P1 = NV_Ith_S(y, 1);
    for (int ii=0; ii<NEQ; ii++) {
        NV_Ith_S(ydot, ii) += -para->duff1 * para->a[ii][1] * P1*P1*P1;
    }

    return(0);
}

static int jac_duffing(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                       void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    SimPara* para = (SimPara*) user_data;

    /* First do the linear part */
    jac_linear(t, y, fy, J, user_data, tmp1, tmp2, tmp3);

    /* Then add the Duffing terms */
    realtype P1 = NV_Ith_S(y, 1);
    for (int ii=0; ii<NEQ; ii++) {
        SM_ELEMENT_D(J, ii, 1) += - 3. * para->duff1 * para->a[ii][1] * P1*P1;
    }

    return(0);
}


static int ode_josephson(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
    SimPara* para = (SimPara*) user_data;

    /* First do the linear part */
    ode_linear(t, y, ydot, user_data);

    /* Then add the Josephson terms */
    realtype P1 = NV_Ith_S(y, 1);
    realtype PHI0 = para->phi0;
    for (int ii=0; ii<NEQ; ii++) {
        NV_Ith_S(ydot, ii) -= para->a[ii][1] * P1;  // remove linear term
        NV_Ith_S(ydot, ii) += para->a[ii][1] * PHI0/(2.*M_PI) * sin(2.*M_PI*P1/PHI0);  // add sine term
    }

    return(0);
}


static int jac_josephson(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                         void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    SimPara* para = (SimPara*) user_data;

    /* First do the linear part */
    jac_linear(t, y, fy, J, user_data, tmp1, tmp2, tmp3);

    /* Then add the Josephson terms */
    realtype P1 = NV_Ith_S(y, 1);
    realtype PHI0 = para->phi0;
    for (int ii=0; ii<NEQ; ii++) {
        SM_ELEMENT_D(J, ii, 1) -= para->a[ii][1];  // remove linear term
        SM_ELEMENT_D(J, ii, 1) += para->a[ii][1] * cos(2.*M_PI*P1/PHI0);  // add cosine term
    }

    return(0);
}


int integrate_cvode(void* user_data,
                    realtype* y0, realtype* tout_arr,
                    realtype* outdata, int nout,
                    realtype reltol, realtype abstol) {
    int flag;
    N_Vector y;
    void* cvode_mem;
    SimPara* para = (SimPara*) user_data;
    /* Calculate internal parameters */
    init_para(para);

    cvode_mem = NULL;
    
    /* Create serial vector of length NEQ for I.C. */
    y = N_VNew_Serial(NEQ);
    if (check_flag((void *)y, "N_VNew_Serial", 0)) return(1);

    /* Initialize y */
    for (int ii=0;ii<NEQ;ii++) {
        NV_Ith_S(y, ii) = y0[ii];
    }

    /* Call CVodeCreate to create the solver memory
     * with stepping and iteration method */
    if (para->stiff_equation==true) {
        cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
    } else {  // nonstiff
        cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
    }
    if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

    /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in y'=f(t,y), the inital time T0, and
    * the initial dependent variable vector y. */
    realtype T0 = tout_arr[0];
    if (para->sys_id == ID_LINEAR) {
        flag = CVodeInit(cvode_mem, ode_linear, T0, y);
    } else if (para->sys_id == ID_DUFFING) {
        flag = CVodeInit(cvode_mem, ode_duffing, T0, y);
    } else if (para->sys_id == ID_JOSEPHSON) {
        flag = CVodeInit(cvode_mem, ode_josephson, T0, y);
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

    /* Create dense SUNMatrix for use in linear solves */
    SUNMatrix A = SUNDenseMatrix(NEQ, NEQ);
    if(check_flag((void *)A, "SUNDenseMatrix", 0)) return(1);

    /* Create dense SUNLinearSolver object for use by CVode */
    SUNLinearSolver LS = SUNDenseLinearSolver(y, A);
    if(check_flag((void *)LS, "SUNDenseLinearSolver", 0)) return(1);

    /* Call CVDlsSetLinearSolver to attach the matrix and linear solver to CVode */
    flag = CVDlsSetLinearSolver(cvode_mem, LS, A);
    if(check_flag(&flag, "CVDlsSetLinearSolver", 1)) return(1);

    /* Set the user-supplied Jacobian routine Jac */
    if (para->sys_id == ID_LINEAR) {
        flag = CVDlsSetJacFn(cvode_mem, jac_linear);
    } else if (para->sys_id == ID_DUFFING) {
        flag = CVDlsSetJacFn(cvode_mem, jac_duffing);
    } else if (para->sys_id == ID_JOSEPHSON) {
        flag = CVDlsSetJacFn(cvode_mem, jac_josephson);
    } else {
        printf("*** UNKNOWN FORCE ID: %d ***\n", para->sys_id);
        return(1);
    }
    if (check_flag(&flag, "CVDlsSetJacFn", 1)) return(1);

    // Set optional inputs
    flag = CVodeSetUserData(cvode_mem, user_data);
    if(check_flag(&flag, "CVodeSetUserData", 1)) return(1);

    realtype tstop=0.0;
    if (para->drive_id==ID_DRIVE_V) {
        para->drive_acc = gsl_interp_accel_alloc();
        para->drive_spline = gsl_spline_alloc(gsl_interp_cspline, nout);
        gsl_spline_init(para->drive_spline, tout_arr, para->drive_V_arr, nout);
        tstop = tout_arr[nout-1];
        flag = CVodeSetStopTime(cvode_mem, tstop);
        if(check_flag(&flag, "CVodeSetUserData", 1)) return(1);
    }
    if (para->add_thermal_noise == true) {
        para->noise1_acc = gsl_interp_accel_alloc();
        // para->noise1_spline = gsl_spline_alloc(gsl_interp_linear, nout);
        para->noise1_spline = gsl_spline_alloc(gsl_interp_cspline, nout);
        gsl_spline_init(para->noise1_spline, tout_arr, para->noise1_arr, nout);
        para->noise0_acc = gsl_interp_accel_alloc();
        // para->noise0_spline = gsl_spline_alloc(gsl_interp_linear, nout);
        para->noise0_spline = gsl_spline_alloc(gsl_interp_cspline, nout);
        gsl_spline_init(para->noise0_spline, tout_arr, para->noise0_arr, nout);
        para->noise2_acc = gsl_interp_accel_alloc();
        // para->noise2_spline = gsl_spline_alloc(gsl_interp_linear, nout);
        para->noise2_spline = gsl_spline_alloc(gsl_interp_cspline, nout);
        gsl_spline_init(para->noise2_spline, tout_arr, para->noise2_arr, nout);
        if (tstop == 0.) {
            tstop = tout_arr[nout-1];
            flag = CVodeSetStopTime(cvode_mem, tstop);
            if(check_flag(&flag, "CVodeSetUserData", 1)) return(1);
        }
    }

    /* In loop, call CVode, write results, and test for error.
     Break out of loop when NOUT preset output times have been reached.  */
    for (int ii=0; ii<NEQ; ii++) {
        outdata[ii] = NV_Ith_S(y,ii);
    }
    realtype tout, tret;
    for(int tt=1; tt<nout; tt++){
        tout = tout_arr[tt];
        /* printf("tout = %g \n", tout); */
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
            // Save result to output vectors
            // OBS: the 1st will be the given initial condition,
            // as in scipy.integrate.odeint
            for (int ii=0; ii<NEQ; ii++) {
                outdata[tt*NEQ + ii] = NV_Ith_S(y,ii);     // P1, flux on node 1
            }
        } else if (flag == CV_TSTOP_RETURN) {
            printf("CVODE: Reached stopping point\n");
            printf("CV_TSTOP_RETURN: tt %d, nout %d, tout %g, tret %g, tstop %g\n", tt, nout, tout, tret, tstop);
            break;
        } else{
            // This shouldn't happen!
            // We already take care of roots
            // and there is check_flag, but you never know...
            printf("*** Something went wrong! ***\n");
            printf("CVODE: Last error flag: %d \n", flag);
            break;
        }
    }
    // printf("DONE: tt %d, nout %d, tout %f, tret %f, tstop %f, dt %f\n", tt, nout, tout*1e9, tret*1e9, tstop*1e9, dt*1e9);

    /* Free y vector */
    N_VDestroy(y);

    /* Free integrator memory */
    CVodeFree(&cvode_mem);

    /* Free the linear solver memory */
    SUNLinSolFree(LS);

    /* Free the matrix memory */
    SUNMatDestroy(A);

    if (para->drive_id==ID_DRIVE_V) {
        gsl_spline_free(para->drive_spline);
        gsl_interp_accel_free(para->drive_acc);
    }
    if (para->add_thermal_noise == true) {
        gsl_spline_free(para->noise1_spline);
        gsl_interp_accel_free(para->noise1_acc);
        gsl_spline_free(para->noise0_spline);
        gsl_interp_accel_free(para->noise0_acc);
        gsl_spline_free(para->noise2_spline);
        gsl_interp_accel_free(para->noise2_acc);
    }

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
