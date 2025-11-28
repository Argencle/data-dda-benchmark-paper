#include <stdio.h>
#include<stdlib.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.h"
//#include "gpbicg_cuda.h"

// #define TEST

// Fonctions auxiliaires pour manipuler les nombres complexes

//cufftDoubleComplex dconjg(cufftDoubleComplex cin);
//double cdabs(cufftDoubleComplex x) ;
//cufftDoubleComplex cdsqrt(cufftDoubleComplex x);


#define  LINE_SIZE 1000
typedef struct {
    cuDoubleComplex pt[LINE_SIZE];
}LINE_OF_COMPLEX;


//-------------------------------------------------------------------------------------------------
int write_line_of_complex_cuda(char * ss, LINE_OF_COMPLEX* d_line, int nou, int ITNO);
//---------------------------------------------------------------


void BICGSTAB_cuda(cuDoubleComplex *X, cuDoubleComplex *XI, cuDoubleComplex *XR, cuDoubleComplex *B, int lda, int nlar, int ndim, 
            int *nou, cuDoubleComplex *WRK_CPP, cuDoubleComplex *ALPHA, cuDoubleComplex *OMEGA, cuDoubleComplex *RHO,
 	    double *NORM, double TOLE, int *ITNO,  int MAXIT, int *STATUS, int *STEPERR
 	     ) {

    cuDoubleComplex zzero = {0.0, 0.0};
    cuDoubleComplex uncomp = {1.0, 0.0};    
    cuDoubleComplex ctmp, ctmp1, ctmp2, KAPPA, BETA, XXI, RHO0;
    double tmp,RESIDU;
    

    cuDoubleComplex *WRK = &WRK_CPP[-lda];    

    LINE_OF_COMPLEX *pt_WR_1 = (LINE_OF_COMPLEX *) &WRK[1 * lda];
    LINE_OF_COMPLEX *pt_WR_2 = (LINE_OF_COMPLEX *) &WRK[2 * lda];
    LINE_OF_COMPLEX *pt_WR_3 = (LINE_OF_COMPLEX *) &WRK[3 * lda];
    LINE_OF_COMPLEX *pt_WR_4 = (LINE_OF_COMPLEX *) &WRK[4 * lda];
    LINE_OF_COMPLEX *pt_WR_5 = (LINE_OF_COMPLEX *) &WRK[5 * lda];
    LINE_OF_COMPLEX *pt_WR_6 = (LINE_OF_COMPLEX *) &WRK[6 * lda];
    //LINE_OF_COMPLEX *pt_WR_7 = (LINE_OF_COMPLEX *) &WRK[7 * lda];
    //    LINE_OF_COMPLEX *pt_WR_8 = (LINE_OF_COMPLEX *) &WRK[8 * lda];
    //    LINE_OF_COMPLEX *pt_WR_9 = (LINE_OF_COMPLEX *) &WRK[9 * lda];
    //    LINE_OF_COMPLEX *pt_WR_10 = (LINE_OF_COMPLEX *) &WRK[10 * lda];
    //    LINE_OF_COMPLEX *pt_WR_11 = (LINE_OF_COMPLEX *) &WRK[11 * lda];
    //    LINE_OF_COMPLEX *pt_WR_12 = (LINE_OF_COMPLEX *) &WRK[12 * lda];

    

     if (*nou == 1) {
        goto label10;
    }
    if (*nou == 2) {
        goto label20;
    }
    if (*nou == 3) {
        goto label30;
    }
    if (*nou == 4){
        goto label100;
    }

//     Set indices for mapping local vectors into wrk
//     IR = 1
//     IRTILDE =2
//     IP = 3
//     IQ = 4
//     IS = 5
//     IT = 6
//     IV = 7
//     IW = 8
//     IZ = 9
//     IXOLD = 10

//     Loop
      *STATUS = 0;
      *STEPERR = -1;
      *NORM=0;
//    1.    r=b-Ax

      C_equal_A_kernel<<< 1+(ndim/1024),1024>>>(&WRK[1*lda], B, ndim);
      C_equal_A_kernel<<< 1+(ndim/1024),1024>>>(X, XI, ndim);      

      *NORM=calc_scalar_product(B,B,ndim).x;		   
      *NORM=sqrt(*NORM);
      printf("NORME  %g    \n", *NORM);
      *nou=1;
      return;

      label10:
      *ITNO = 0;

//     2. rtilde=r, 3. p=v=0


//       C_equal_A_kernel<<< 1+(ndim/1024),1024>>>(&WRK[8*lda], XR, ndim);
       C_equal_C_minus_A<<<1+ndim/1024,1024   >>>( &WRK[1 * lda], XR,ndim );       
       C_equal_A_kernel<<< 1+(ndim/1024),1024>>>(&WRK[2*lda], &WRK[1*lda], ndim);       
       C_equal_zero_kernel<<<1+ndim/1024,1024   >>>( &WRK[ 3 * lda],ndim);
       C_equal_zero_kernel<<<1+ndim/1024,1024   >>>( &WRK[ 4 * lda],ndim);
       tmp=calc_scalar_product ( &WRK[ 1 * lda],&WRK[ 1 * lda], ndim ).x;

//      write(*,*) 'INIT RESIDU',dsqrt(TMP)/NORM,'NORM',norm
//     4. rho=alpha=omega=1
      *RHO = uncomp;
      *ALPHA = uncomp;
      *OMEGA = uncomp;

      label100:

      (*ITNO)++;
    
//     5. rho=dot(rtilde,r)

       RHO0 = *RHO;
//             printf("RHO0  %g    \n", RHO0);		        
       *RHO = calc_scalar_product ( &WRK[ 2 * lda],&WRK[ 1 * lda], ndim );

//     6. beta=rho*alpha/(rho0*omega)
      KAPPA = cuCmul(RHO0,*OMEGA);

      if (cdabs(KAPPA) == 0.0) {
         *STATUS = -3;
         *STEPERR = 6;
         return;
	 }	 

      BETA = cuCdiv(cuCmul(*RHO,*ALPHA),KAPPA);
// printf("BETA  %g    \n", BETA);
// printf("RHO  %g    \n", *RHO);
// printf("ALPHA  %g    \n", *ALPHA);	  
//printf("KAPPA  %g    \n", KAPPA);
//printf("OMEGA  %g    \n", *OMEGA);	 
//     7. p=r+beta*(p-omega*v), 8. v=Q1AQ2p
//        write_line_of_complex_cuda("WRK1", pt_WR_1, *nou, *ITNO);
//        write_line_of_complex_cuda("WRK3", pt_WR_3, *nou, *ITNO);
//        write_line_of_complex_cuda("WRK7", pt_WR_7, *nou, *ITNO);		

       calc_WRK3_BICGSTAB<<< 1+(ndim/1024),1024>>>(WRK, BETA, *OMEGA, lda, ndim);
       C_equal_A_kernel<<< 1+(ndim/1024),1024>>>( XI, &WRK[3*lda], ndim);
//       write_line_of_complex_cuda("WRK3", pt_WR_3, *nou, *ITNO);
      *nou=2;
      return;

      label20:		

//     9. xi=dot(rtilde,v)

       C_equal_A_kernel<<< 1+(ndim/1024),1024>>>( &WRK[4*lda], XR, ndim);
       XXI=calc_scalar_product(&WRK[2*lda],&WRK[4*lda],ndim);
//       write_line_of_complex_cuda("WRK7", pt_WR_7, *nou, *ITNO);
//       printf("XXI  %g    \n", XXI);
       
//     10. alpha=rho/xi
      if (cdabs(XXI) == 0.0) {
         *STATUS = -3;
         *STEPERR = 10;
         return;
	 }

      *ALPHA = cuCdiv(*RHO,XXI);
      KAPPA = zzero;
      
//     11. s=r-alpha*v, 12. if ||s||<breaktol then soft-breakdown has occurred

       
       calc_WRK5_BICGSTAB<<< 1+(ndim/1024),1024>>>(WRK, *ALPHA, lda , ndim);
       KAPPA=calc_scalar_product(&WRK[5*lda],&WRK[5*lda],ndim);
       KAPPA=cdsqrt(KAPPA);
//         printf("KAPPA  %g    \n", KAPPA);
//	  write_line_of_complex_cuda("WRK5", pt_WR_5, *nou, *ITNO);
	  
      if (cdabs(KAPPA) < 1.e-15) {
         *STATUS = -2;		  
         *STEPERR = 12;
	 C_equal_A_kernel<<< 1+(ndim/1024),1024>>>(X , &WRK[3*lda], ndim);
	 return;
	 }

//     13. t=Q1AQ2s

       C_equal_A_kernel<<< 1+(ndim/1024),1024>>>(XI , &WRK[5*lda], ndim);
//      write_line_of_complex_cuda("WRK5bis", pt_WR_5, *nou, *ITNO); 	   
      *nou=3;
      return;

      label30:

//     14. omega=dot(t,s)/dot(t,t)
       C_equal_A_kernel<<< 1+(ndim/1024),1024>>>(&WRK[ 6 * lda] , XR, ndim);	
       ctmp1=calc_scalar_product ( &WRK[ 6 * lda],&WRK[ 6 * lda], ndim );
       ctmp2=calc_scalar_product ( &WRK[ 6 * lda],&WRK[ 5 * lda], ndim );
//       write_line_of_complex_cuda("WRK6", pt_WR_6, *nou, *ITNO);
      
      if (cdabs(ctmp1) == 0.0) {
         *STATUS = -3;
         *STEPERR = 14;
	 return;
	 }

       *OMEGA = cuCdiv(ctmp2,ctmp1);
//	printf("OMEGA  %g    \n", *OMEGA);
//     15. x=x+alpha*p+omega*s, 16. r=s-omega*t


       
       calc_X_BICGSTAB<<< 1+(ndim/1024),1024>>>(WRK, X, *ALPHA, *OMEGA, lda, ndim);
       calc_WRK1_BICGSTAB<<< 1+(ndim/1024),1024>>>(WRK, *OMEGA, lda, ndim);   			  
       tmp=calc_scalar_product ( &WRK[ 1 * lda],&WRK[ 1 * lda], ndim ).x;
       
       tmp=sqrt(tmp)/ *NORM;
      
//printf("RESIDU %g\nIteration: %d\n", tmp, *ITNO);
printf("RESIDU %.17g, Iteration: %d\n", tmp, *ITNO);
//	printf("TOLE  %g    \n", TOLE);	
//     17. check stopping criterion
       if (*ITNO > MAXIT) {
         *STATUS = -1;
         *ITNO = MAXIT;
	 return;
	 }
      if (tmp < TOLE) {
         *STATUS=1;
         *nou=4;
         return;
	 }

      goto label100;

      return;
}
