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
#include <gsl/gsl_spline.h>

// #define DEBUG

/* Program constants */
#define true  1
#define false 0
#define NEQ   4              // number of equations

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
#define ID_DRIVE_dVdt 3

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
    realtype Cl;  // F
    realtype Cr;  // F
    realtype R1;  // ohm
    realtype L1;  // H
    realtype C1;  // F
    realtype R2;  // ohm
    realtype L2;  // H
    realtype C2;  // F

    /* Drive */
    int drive_id;  // ID_LOCKIN, ID_DRIVE_V or ID_DRIVE_dVdt
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
    int sys_id;  // ID_LINEAR, ID_DUFFING or ID_JOSEPHSON

    /* Duffing: V/L * (1. - duff * V*V) */
    realtype duff;  // V^-2

    /* Josephson */
    realtype phi0;

    /* Thermal noise */
    int add_thermal_noise;  // bool
    realtype* noise1_arr;  // V
    realtype* noise2_arr;  // V
    gsl_spline* noise1_spline;  // internal
    gsl_interp_accel* noise1_acc;  // internal
    gsl_spline* noise2_spline;  // internal
    gsl_interp_accel* noise2_acc;  // internal
} SimPara;

/* Functions Called by the Solver */
static realtype V_dot_drive(realtype t, void* user_data);
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

static realtype V_dot_drive(realtype t, void* user_data) {
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
            ret += -w_arr[i] * A_arr[i] * sin(w_arr[i]*t + P_arr[i]);
        }

        return ret;
    } else if (drive_id==ID_DRIVE_V) {
        return gsl_spline_eval_deriv(para->drive_spline, t, para->drive_acc);
    } else { // drive_id==ID_DRIVE_dVdt
        // printf("%f - %f - %f\n", t * 1e9, para->drive_spline->interp->xmin * 1e9, para->drive_spline->interp->xmax * 1e9);
        return gsl_spline_eval(para->drive_spline, t, para->drive_acc);
    }
}

static realtype V_noise(realtype t, int node, void* user_data) {
    SimPara* para = (SimPara*) user_data;
    if (node == 1){
        return gsl_spline_eval(para->noise1_spline, t, para->noise1_acc);
    } else {
        return gsl_spline_eval(para->noise2_spline, t, para->noise2_acc);
    }
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


static int ode_josephson(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
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

    realtype PHI0 = para->phi0;

    NV_Ith_S(ydot, 0) = V1;
    NV_Ith_S(ydot, 1) = g0 * (-P1 / (Csum1 * L1) - (V1+Vn1) / (Csum1 * R1) + Cl / Csum1 * V0_dot + g1 * (-sin(2.*M_PI*P2/PHI0)*PHI0/(2.*M_PI*Csum2*L2) - (V2+Vn2) / (Csum2 * R2)));
    NV_Ith_S(ydot, 2) = V2;
    NV_Ith_S(ydot, 3) = g0 * (-sin(2.*M_PI*P2/PHI0)*PHI0/(2.*M_PI*Csum2*L2) - (V2+Vn2) / (Csum2 * R2) + g2 * (-P1 / (Csum1 * L1) - (V1+Vn1) / (Csum1 * R1) + Cl / Csum1 * V0_dot));

    return(0);
}


static int jac_josephson(long int N, realtype t,
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

    realtype Csum1 = C1 + Cl + Cr;
    realtype Csum2 = C2 + Cr;
    realtype g0 = 1. / (1. - Cr*Cr / (Csum1 * Csum2));
    realtype g1 = Cr / Csum1;
    realtype g2 = Cr / Csum2;

    realtype PHI0 = para->phi0;

    DENSE_ELEM(J, 0, 0) = RCONST(0.);
    DENSE_ELEM(J, 0, 1) = RCONST(1.);
    DENSE_ELEM(J, 0, 2) = RCONST(0.);
    DENSE_ELEM(J, 0, 3) = RCONST(0.);

    DENSE_ELEM(J, 1, 0) = -g0 / (Csum1 * L1);
    DENSE_ELEM(J, 1, 1) = -g0 / (Csum1 * R1);
    DENSE_ELEM(J, 1, 2) = -g0 * g1 / (Csum2 * L2) * cos(2.*M_PI*P2/PHI0);
    DENSE_ELEM(J, 1, 3) = -g0 * g1 / (Csum2 * R2);

    DENSE_ELEM(J, 2, 0) = RCONST(0.);
    DENSE_ELEM(J, 2, 1) = RCONST(0.);
    DENSE_ELEM(J, 2, 2) = RCONST(0.);
    DENSE_ELEM(J, 2, 3) = RCONST(1.);

    DENSE_ELEM(J, 3, 0) = -g0 * g2 / (Csum1 * L1);
    DENSE_ELEM(J, 3, 1) = -g0 * g2 / (Csum1 * R1);
    DENSE_ELEM(J, 3, 2) = -g0 / (Csum2 * L2) * cos(2.*M_PI*P2/PHI0);
    DENSE_ELEM(J, 3, 3) = -g0 / (Csum2 * R2);

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

    /* Call CVDense to specify the CVDENSE dense linear solver */
    flag = CVDense(cvode_mem, NEQ);
    if (check_flag(&flag, "CVDense", 1)) return(1);

    /* Set the Jacobian routine (user-supplied) */
    if (para->sys_id == ID_LINEAR) {
        flag = CVDlsSetDenseJacFn(cvode_mem, jac_linear);
    } else if (para->sys_id == ID_DUFFING) {
        flag = CVDlsSetDenseJacFn(cvode_mem, jac_duffing);
    } else if (para->sys_id == ID_JOSEPHSON) {
        flag = CVDlsSetDenseJacFn(cvode_mem, jac_josephson);
    } else {
        printf("*** UNKNOWN FORCE ID: %d ***\n", para->sys_id);
        return(1);
    }
    if (check_flag(&flag, "CVDlsSetDenseJacFn", 1)) return(1);

    // Set optional inputs
    flag = CVodeSetUserData(cvode_mem, user_data);
    if(check_flag(&flag, "CVodeSetUserData", 1)) return(1);

    realtype tstop=0.0;
    if (para->drive_id==ID_DRIVE_V || para->drive_id==ID_DRIVE_dVdt) {
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
    outdata[0] = NV_Ith_S(y,0);  // P1, flux on node 1
    outdata[1] = NV_Ith_S(y,1); // V1, voltage on node 1
    outdata[2] = NV_Ith_S(y,2); // P2, flux on node 2
    outdata[3] = NV_Ith_S(y,3); // V2, voltage on node 2
    int ii;
    realtype tout, tret;
    for(ii = 1; ii < nout; ii++){
        tout = tout_arr[ii];

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
            outdata[ii*NEQ] = NV_Ith_S(y,0);     // P1, flux on node 1
            outdata[ii*NEQ + 1] = NV_Ith_S(y,1); // V1, voltage on node 1
            outdata[ii*NEQ + 2] = NV_Ith_S(y,2); // P2, flux on node 2
            outdata[ii*NEQ + 3] = NV_Ith_S(y,3); // V2, voltage on node 2
        } else if (flag == CV_TSTOP_RETURN) {
            printf("CVODE: Reached stopping point\n");
            printf("CV_TSTOP_RETURN: ii %d, nout %d, tout %g, tret %g, tstop %g\n", ii, nout, tout, tret, tstop);
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
    // printf("DONE: ii %d, nout %d, tout %f, tret %f, tstop %f, dt %f\n", ii, nout, tout*1e9, tret*1e9, tstop*1e9, dt*1e9);

    /* Free y vector */
    N_VDestroy_Serial(y);

    /* Free integrator memory */
    CVodeFree(&cvode_mem);

    if (para->drive_id==ID_DRIVE_V || para->drive_id==ID_DRIVE_dVdt) {
        gsl_spline_free(para->drive_spline);
        gsl_interp_accel_free(para->drive_acc);
    }
    if (para->add_thermal_noise == true) {
        gsl_spline_free(para->noise1_spline);
        gsl_interp_accel_free(para->noise1_acc);
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
