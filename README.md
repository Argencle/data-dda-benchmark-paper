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

2. I let the `FFTW_ESTIMATE`:
```bash
grep -R --line-number 'FFTW_ESTIMATE=64' .
```

```bash
find ./cdmlib -type f \( -name '*.f' -o -name '*.F' -o -name '*.f90' -o -name '*.F90' \) -exec sed -Ei 's/^([[:space:]]*FFTW_ESTIMATE[[:space:]]*=[[:space:]]*)64([[:space:]]*.*)$/\10\2/' {} +
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