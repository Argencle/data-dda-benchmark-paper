# Modifications to do on the codes

## On my laptop
### DDSCAT
1. Create the `diel` source files (`m1.50_0.00.dat` and `m1.313_0.00.dat`)
2. Put the `mkl_dfti.f90` file in the `src` folder
3. Set or copy paste the modified `Makefile`, `alphadiad.f90`, `cisi.f90`, `getfml.f90`, `cgcommon.f90`, `cxfft3_mkl.f90` files
4. In `getfml.f90`, `cgcommon.f90`, `gpbicg.f90`, `qmrpim2.f90 `, `zbcg2wp.f90` the following modifications where made to display more digits and the number of iterations and MVPs:
```
> !*** diagnostic
>                write(0,*) ' GPBICG: ITERN=',ITERN, &
>                      ' MULTIPLICATIONS=',MULTIPLICATIONS
906c908,909
< 
---
>                write(0,*) ' QMRCCG: ITERN=',ITERN, &
>                      ' MULTIPLICATIONS=',MULTIPLICATIONS
948a952
>                write(0,*) 'BICGSTAB: ITERN', IPAR(11)
1030c1034
<             WRITE(IDVOUT,FMT='(A,1PE11.4,A,1PE11.4)')       &
---
>             WRITE(IDVOUT,FMT='(A,1PE24.16,A,1PE24.16)')       &
```

```
281,282c281,282
< 9000  FORMAT(1X,'iter= ',I5,' frac.err= ',0P,F11.7)
< 9001  FORMAT(1X,'iter= ',I5,' frac.err= ',0P,F11.7,' min.err=',0P,F11.7, &
---
> 9000  FORMAT(1X,'iter= ',I5,' frac.err= ',0P,F25.16)
> 9001  FORMAT(1X,'iter= ',I5,' frac.err= ',0P,F25.16,' min.err=',0P,F11.7, &
```

```
251c251
<            WRITE(CMSGNM,FMT='(A,I8,A,1P,E10.3)') &
---
>            WRITE(CMSGNM,FMT='(A,I8,A,1P,E25.16)') &
```

```
104c104
<          WRITE(CMSGNM,FMT='(A,I8,A,1P,E10.3)') &
---
>          WRITE(CMSGNM,FMT='(A,I8,A,1P,E25.16)') &
```

```
688c688
<          WRITE(CMSGNM,FMT='(A,I8,A,1P,E10.3)') &
---
>          WRITE(CMSGNM,FMT='(A,I8,A,1P,E25.16)') &
```

5. Install `ddscatcli`:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ddscatcli
```
6. Set up `ddscatcli`:
```bash
export DDSCAT_PAR=ddscat7.3.4_250505/ddscat.par
export DDSCAT_EXE=/home/argentic@coria.fr/Bureau/Work/paper1/ddscat7.3.4_250505/src/ddscat
```

### IFDDA
v1.0.26
1. Set or copy paste the modified `bicgstab_cuda.cu` and `bicgstab_cuda_simp.cu` to display more digits and Makefiles

```
223c223
<     printf("RESIDU %g, Iteration: %d\n", tmp, *ITNO);
---
>     printf("RESIDU %.17g, Iteration: %d\n", tmp, *ITNO);
```

```
214c14
<     printf("RESIDU %g, Iteration: %d\n", tmp, *ITNO);
---
>     printf("RESIDU %.17g, Iteration: %d\n", tmp, *ITNO);
```

2. Change the the `FFTW_ESTIMATE` parameters:
```bash
grep -R --line-number 'FFTW_ESTIMATE=64' .
```

```bash
find ./cdmlib -type f \( -name '*.f' -o -name '*.F' -o -name '*.f90' -o -name '*.F90' \) -exec sed -Ei 's/^([[:space:]]*FFTW_ESTIMATE[[:space:]]*=[[:space:]]*)64([[:space:]]*.*)$/\10\2/' {} +
```

3. Copy-paste the `main.f` file in `test/test_command/` to modify the `cmplx` functions especially for `epsmulti`:
```
199c199
<          E0m(i)    = (0.d0, 0.d0)
---
>          E0m(i)    = cmplx(0.d0, 0..d0)
332c332
<                         E0m(ipwave) = temp_real+(0.d0,1.d0)*temp_imag
---
>                         E0m(ipwave) = cmplx(temp_real, temp_imag)
1115,1116c1115
< !     epsmulti(ieps) = cmplx(epsreal, epsimag)
<                   epsmulti(ieps) = epsreal+(0.d0,1.d0)* epsimag
---
>                   epsmulti(ieps) = cmplx(epsreal, epsimag)
1190,1198c1190,1198
<                   epsanimulti(1,1,1) = (2.D0, 0.D0)
<                   epsanimulti(2,1,1) = (0.D0, 0.D0)
<                   epsanimulti(3,1,1) = (0.D0, 0.D0)
<                   epsanimulti(1,2,1) = (0.D0, 0.D0)
<                   epsanimulti(2,2,1) = (2.D0, 0.D0)
<                   epsanimulti(3,2,1) = (0.D0, 0.D0)
<                   epsanimulti(1,3,1) = (0.D0, 0.D0)
<                   epsanimulti(2,3,1) = (0.D0, 0.D0)
<                   epsanimulti(3,3,1) = (2.D0, 0.D0)
---
>                   epsanimulti(1,1,1) = cmplx(2.D0, 0.D0)
>                   epsanimulti(2,1,1) = cmplx(0.D0, 0.D0)
>                   epsanimulti(3,1,1) = cmplx(0.D0, 0.D0)
>                   epsanimulti(1,2,1) = cmplx(0.D0, 0.D0)
>                   epsanimulti(2,2,1) = cmplx(2.D0, 0.D0)
>                   epsanimulti(3,2,1) = cmplx(0.D0, 0.D0)
>                   epsanimulti(1,3,1) = cmplx(0.D0, 0.D0)
>                   epsanimulti(2,3,1) = cmplx(0.D0, 0.D0)
>                   epsanimulti(3,3,1) = cmplx(2.D0, 0.D0)
1208,1209c1208,1209
<                         epsanimulti(iraw, icol, 1) =epsanimultireal
<      $                       +(0.d0,1.d0) *epsanimultiimag
---
>                         epsanimulti(iraw, icol, 1) =
>      $                        cmplx(epsanimultireal, epsanimultiimag)
1229,1230c1229
<      $                          epsanimultireal +(0.d0,1.d0)
<      $                          *epsanimultiimag
---
>      $                           cmplx(epsanimultireal, epsanimultiimag) 
```

### ADDA
1. Set or copy paste the modified `const.h` file:
```
147,148c147,148
< #define EFORM "%.10E"             // fixed width
< #define GFORM "%.10g"             // variable width (showing significant digits)
---
> #define EFORM "%.17E"             // fixed width
> #define GFORM "%.17g"             // variable width (showing significant digits)
```

2. Modify the GPFA file `cfft99D.f` to actually use double precision on the constants:
```
527,529c527,529
<       DATA SIN36/0.587785252292473/,COS36/0.809016994374947/,
<      *     SIN72/0.951056516295154/,COS72/0.309016994374947/,
<      *     SIN60/0.866025403784437/
---
>       DATA SIN36/0.587785252292473D0/,COS36/0.809016994374947D0/,
>      *     SIN72/0.951056516295154D0/,COS72/0.309016994374947D0/,
>      *     SIN60/0.866025403784437D0/
```

#### setting up OCL_BLAS
1. Install clBLAS
```bash
git clone https://github.com/clMathLibraries/clBLAS.git
cd clBLAS/src/
cmake -DCMAKE_INSTALL_PREFIX=$HOME -DCMAKE_BUILD_TYPE=Release .
make 
make install/fast
```
2. Compile:
```bash
make ocl OPTIONS=OCL_BLAS EXTRA_FLAGS="-I$HOME/./include -L$HOME/lib64 -Wl,-rpath,$HOME/lib64"
```
where `-Wl,-rpath,$HOME/lib64` adds `/home/…/lib64` to the binary's search table (equivalent to setting `export LD_LIBRARY_PATH=`).
3. Rename `adda_ocl` as `adda_ocl_blas`.


## On the cluster (CRIANN)
### DDSCAT
1. Modify the `Makefile` and load `module load gcc-native/12.2`
```
PRECISION       = dp
CXFFTMKL.f      = $(MKL_f)
CXFFTMKL.o      = $(MKL_o)
MKLM            = $(MKL_m)
DOMP            = -Dopenmp
OPENMP          = -fopenmp
MPI.f           = mpi_fake.f90
MPI.o           = mpi_fake.o
DMPI            =
FC              = ftn
FFLAGS          = -O2
LFLAGS          = -L/soft/intel/oneapi/mkl/2023.0.0/lib/intel64 \
                  -Wl,-rpath,/soft/intel/oneapi/mkl/2023.0.0/lib/intel64 \
                  -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
```
2. Copy paste the modified files.
3. Create a python enviroment (I tried with the default Python 3.9.18) and install `ddscatcli`
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ddscatcli
```
4. Set the export in the job file.

### IFDDA
1. Copy the modified files and Makefile.
2. Load the modules:
- MPI:
```bash
module load gcc-native/12.2
module load cray-fftw cray-hdf5
```
- GPU
```bash
module load gcc-native/12.2
module load cray-fftw cray-hdf5
module load nvhpc-hpcx-cuda12/23.11
```
3. Redo the same for `FFTW_ESTIMATE=0` but modified the Makefile to add `_measure` at the end of the executables to differentiate them
 
### ADDA
1. Copy the modified file.
2. Load the modules and compile:
- SEQ and MPI
```bash
module load gcc-native/12.2
module load cray-fftw
make CC=cc CF=ftn seq
make CC=cc CF=ftn MPICC=cc mpi
```
- GPU
```bash
module load gcc-native/12.2
module load cray-fftw
module load nvhpc-byo-compiler
module load math/clFFT/2.14.0-nvidia
make CC=cc CF=ftn EXTRA_FLAGS="-I$NVHPC_ROOT/cuda/12.3/include -L$NVHPC_ROOT/cuda/12.3/lib64/" ocl
```
3. Dowload and install `clBLAS`:
```bash

```
4. Compile
```bash
make CC=cc FC=ftn OPTIONS=OCL_BLAS ocl EXTRA_FLAGS="-I$NVHPC_ROOT/cuda/include -I$HOME/.local/include -L$NVHPC_ROOT/cuda/lib64 -L$HOME/.local/lib64 -Wl,-rpath,$NVHPC_ROOT/cuda/lib64 -Wl,-rpath,$HOME/.local/lib64 -lOpenCL -lclBLAS"
```
5. rename the executable as `adda_ocl_blas`