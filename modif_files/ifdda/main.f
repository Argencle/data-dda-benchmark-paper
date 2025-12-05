      PROGRAM test_cdmlib
#ifdef USE_HDF5
      use HDF5
#endif

      implicit none
c     integer
      integer ii,jj,kk,i,j,k,nstop
      integer  nlocal,nmacro,nsection,nsectionsca,nforce ,nforced
     $     ,ntorque,ntorqued,nsens,nproche,nlecture,nquickdiffracte,nrig
     $     ,nquad,nenergie,nmat,nprecon,ninitest

c     variables for the object
      integer nbsphere3,nbsphere,ndipole,IP(3),test,numberobjetmax
     $     ,numberobjet,ierror
      parameter (numberobjetmax=3)
      integer nx,ny,nz,nx2,ny2,nxy2,nz2,nxm,nym,nzm,nxmp,nymp,nzmp
     $     ,ntotal,nmaxpp,nxym1,nzm1,nzms1
      integer subunit,nphi,ntheta

c     definition of the size for the code
      integer nmax, ntotalm

c     variables for the positions
      integer ng,speckseed
      double precision x0,y0,z0,x,y,z,xx0,yy0,zz0,rayon,density,side
     $     ,sidex,sidey,sidez,hauteur,xgmulti(numberobjetmax)
     $     ,ygmulti(numberobjetmax) ,zgmulti(numberobjetmax)
     $     ,rayonmulti(numberobjetmax),demiaxea ,demiaxeb,demiaxec
     $     ,thetaobj,phiobj,psiobj,lc,hc
      double precision aretecube
      double precision, allocatable :: xs(:),ys(:),zs(:)
      double precision, allocatable :: xswf(:),yswf(:),zswf(:)
      double precision pi,lambda,lambda10n,k0,k03,epi,epr,c

c     variables for the material
      double precision eps0
      double complex, allocatable :: polarisa(:,:,:),epsilon(:,:,:)
      double complex epsmulti(numberobjetmax)
     $     ,epsanimulti(3,3,numberobjetmax)
      character(2) polarizability
      character (64), dimension(numberobjetmax) :: materiaumulti
      character(64) materiau,object,beam,namefileobj,namefileinc
     $     ,filereread
      character(3) trope

c     variables for the incident field and local field
      double precision, allocatable :: incidentfield(:),localfield(:)
      double precision, allocatable :: macroscopicfield(:)
      double precision, allocatable :: forcex(:),forcey(:),forcez(:)
      double precision, allocatable :: torquex(:),torquey(:),torquez(:)
      double precision forcexmulti(numberobjetmax)
     $     ,forceymulti(numberobjetmax),forcezmulti(numberobjetmax)
     $     ,torquexmulti(numberobjetmax),torqueymulti(numberobjetmax)
     $     ,torquezmulti(numberobjetmax)
      double precision ss,pp,theta,phi,I0
      integer nbinc
      double precision thetam(10), phim(10), ppm(10), ssm(10)
      double complex E0m(10)
      double complex Eloc(3),Em(3),E0,uncomp,icomp,zzero
      double complex, allocatable :: macroscopicfieldx(:)
      double complex, allocatable :: macroscopicfieldy(:)
      double complex, allocatable :: macroscopicfieldz(:)
      double complex, allocatable :: localfieldx(:)
      double complex, allocatable :: localfieldy(:)
      double complex, allocatable :: localfieldz(:)
      double complex, allocatable :: incidentfieldx(:)
      double complex, allocatable :: incidentfieldy(:)
      double complex, allocatable :: incidentfieldz(:)
      double complex propaesplibre(3,3)
      double complex, allocatable :: FF(:),FF0(:),FFloc(:)
c      ,FFprecon
c      double complex, dimension(3*nzm,3*nzm,nxm*nym) :: sdetnn
      double complex , dimension (:,:,:),  allocatable :: sdetnn
      double complex , dimension (:),  allocatable :: FFprecon

c     Green function
      integer, allocatable :: Tabdip(:),Tabmulti(:)
      integer indice
      double complex, allocatable :: FFTTENSORxx(:),
     $     FFTTENSORxy(:),FFTTENSORxz(:),FFTTENSORyy(:),FFTTENSORyz(:),
     $     FFTTENSORzz(:),vectx(:),vecty(:),vectz(:)

      double precision forcet(3),forcem,forcemie
      double precision couplet(3),couplem
      double complex Eder(3,3)

c     computation of the cross section
      integer iphi,itheta
      double precision MIECEXT,MIECABS,MIECSCA ,GSCA,Cext,normal(3)
     $     ,deltatheta,deltaphi,Csca,Cscai,Cabs,gasym,thetas,phis
     $     ,efficacite,efficaciteref,efficacitetrans
      double complex ctmp

c     variables for the iterative method
      integer ldabi, nlar
      integer nnnr,ncompte
      integer NLIM,ndim,nou,maxit,nstat,nloop,STEPERR
      double precision  NORM,TOL,norm1,norm2,tolinit,tol1,tempsreelmvp
     $     ,tempsreeltotal
      double complex ALPHA,BETA,GPETA,DZETA,R0RN

c     COMMON /ONTHEHEAP/ b,xr,xi,wrk
      double complex, allocatable :: xr(:),xi(:)
      double complex, allocatable :: wrk(:,:)

c     double complex wrk(*), xi(*), xr(*), b(*)
c     POINTER ( xr_p, xr ), ( b_p, b )
c     POINTER ( wrk_p, wrk ), ( xi_p, xi)

c     Poynting vector
      integer nr,nrmax,nw,nwmax
      double precision Poyntinginc

c     Info string
      character(64) infostr

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     nouvelle variable a passer en argument d'entree
c     power et diametre
      double precision P0,w0,xgaus,ygaus,zgaus,quatpieps0
      character(12) methodeit

c     nouvelle variable de sortie Irra
      double precision irra


c     nouvelle variable
      integer nloin

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Creation des nouvelles variables
      integer na

c     variable pour avoir l'image a travers la lentille
      integer nlentille,nobjet,nfft2d,nfft2d2,nquicklens,ntypemic,nside
     $     ,npoynting
      double precision, allocatable :: thetafield(:),phifield(:)
      double precision, allocatable :: poyntingfield(:)
      double precision, allocatable :: poyntingfieldpos(:)
      double precision, allocatable :: poyntingfieldneg(:)
      integer, allocatable :: tabfft2(:)
      double precision kx,ky,kz,deltakx,deltaky,numaper,deltax,gross
     $     ,numaperinc,numaperinc2,kcnax,kcnay,zlens,deltapoyntingx
     $     ,deltapoyntingy,numaperil,psiinc,zstep,zinf,zsup
      double precision, allocatable :: kxy(:),xy(:),kxypoynting(:)
      double complex, allocatable :: Eimagex(:),Eimagey(:)
     $     ,Eimagez(:),Eimageincx(:)
     $     ,Eimageincy(:) ,Eimageincz(:)
     $     ,Efourierx(:) ,Efouriery(:)
     $     ,Efourierz(:),Efourierincx(:)
     $     ,Efourierincy(:) ,Efourierincz(:)
     $     ,Ediffkzpos(:,:,:) ,Ediffkzneg(:,:,:)

      character(len=100) :: h5file

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     New variables for command-line arguments
      integer :: nargs, iargs
      character(len=100) :: arg, argval
      double precision :: epsreal, epsimag,
     $      epsanimultireal, epsanimultiimag
      integer :: alloc_size
      integer :: ntemp, ipwave, iprop, iobj, ipos, ieps, iraw, icol,
     $      itoken, param_count
      double precision :: temp_real, temp_imag
      integer GET_PARAM_COUNT
      external GET_PARAM_COUNT

c********************************************************
c     Initialise default values (only those needed if the user does not
c     pass command-line arguments to modify them)
c********************************************************

c     constant
      c = 299792458.d0
      quatpieps0 = 1.d0/(c*c*1.d-7)
      pi = dacos(-1.d0)
      icomp = (0.d0,1.d0)

c     DATA INPUT
      lambda = 632.8d0            ! wavelength
      P0 = 1.d0                   ! power
      w0 = lambda*10.d0           ! waist

C     Default values for beam
      beam = 'pwavelinear'
      theta = 0.d0
      phi   = 0.d0
      pp    = 0.d0
      ss    = 1.d0

      nbinc = 1
      do 1 i = 1, 10
         thetam(i) = 0.d0
         phim(i)   = 0.d0
         ppm(i)    = 0.d0
         ssm(i)    = 1.d0
         E0m(i)    = (0.d0, 0.d0)
1     continue

c     Default values for object
      object = 'sphere'
      numberobjet = 1
      trope = 'iso'
      nnnr = 10
      nxm = nnnr
      nym = nnnr
      nzm = nnnr
      thetaobj = 0.d0
      phiobj = 0.d0
      psiobj = 0.d0
      do 2 i = 1, numberobjetmax
         xgmulti(i) = 0.0d0
         ygmulti(i) = 0.0d0
         zgmulti(i) = 0.0d0
         if (i .eq. 1) then
            rayonmulti(i) = 100.0d0
            epsmulti(i) = (1.1d0, 0.0d0)
         else
            rayonmulti(i) = 0.0d0
            epsmulti(i) = (0.0d0, 0.0d0)
         end if
         materiaumulti(i) = 'xx'
2     continue

c     ng = Sphere seed
c     aretecube = meshsize
c     pp = Polarization TM(1) TE(0)
c     ss = polarization Right(1) Left(-1)

c     Default values for iterative solver
      nlar = 12
      methodeit = 'GPBICG1'
      nlim = 1000
      ninitest = 1
      nprecon = 0
      tolinit = 1.d-4
      nlecture = 0
      filereread ='toto'          ! name fo the file if reread the local field.

      nrig = 0
      polarizability = 'RR'

c     Optical Force
      nforce = 0                  ! (0) do not compute or (1) compute the optical force.
      nforced = 0                 ! (0) do not compute or (1) compute the density of optical force.
      ntorque = 0                 ! (0) do not compute or (1) compute the optical torque.
      ntorqued = 0                ! (0) do not compute or (1) compute the density of optical torque.

c     Far Field
      nsection = 1                ! 0 do not compute the cross section, 1 compute the cross section.
      nsectionsca = 0             !1: calcul C_sca, Poynting and g with radiating dipole.
      nenergie = 0                ! 0 do not compute energy, 1 compute energy conservation.
      nquickdiffracte = 0         ! 0 compute far field classically, 1 compute far field with FFT.
      nfft2d = 128
      nphi = 72
      ntheta = 36

c     Near Field
      nproche = 0
      nxmp = 0                    ! if nproche=2 used then the addsize along x : nx+2*nxmp
      nymp = 0                    ! if nproche=2 used then the addsize along y : ny+2*nymp
      nzmp = 0                    ! if nproche=2 used then the addsize along z : nz+2*nzmp
      nlocal = 0                  ! 0 do not compute the local field, 1 compute the local field
      nmacro = 0                  ! 0 do not compute the macroscopic field, 1 compute the macroscopic field

c     Only dipoles
      nobjet = 0                  ! 1 compute only the position of the
                                  ! dipole, all the other options are
                                  ! disabled.

C     Interaction Term
      nquad = 0                   !0 -> 5 define the level of integration of the Green tensor.

C     Save Data
      nmat = 1                    ! 1 do not save the data, 0 save the data in mat file, 2 save the data in one hdf5 file.
      h5file = 'ifdda.h5'         ! name of the hdf5 file

c********************************************************
c     Define the command-line args
c********************************************************

c     Get the number of arguments passed (excluding the program name)
      nargs = IARGC()
      iargs = 1

100   if (iargs .gt. nargs) go to 200
         call getarg(iargs, arg)
c        beam argument
         if (trim(arg) .eq. '-beam') then
            iargs = iargs + 1
            if (iargs .gt. nargs) then
               write(*,*) 'Error: -beam without type'
               stop
            end if
            call getarg(iargs, beam)

            param_count = GET_PARAM_COUNT(iargs+1, nargs)

            if (beam(1:11) .eq. 'pwavelinear') then
C              for pwavelinear, optionally read pp and ss
               if (param_count .eq. 2) then
                  call getarg(iargs+1, argval)
                  read(argval,*) pp
                  call getarg(iargs+2, argval)
                  read(argval,*) ss
                  iargs = iargs + 2
               else if (param_count .eq. 0) then
                  write(*,*) 'using default values for pwavelinear'
               else
                  write(*,*) 'Error: requires 0 or 2 parameters'
                  stop
               end if
            else if (beam(1:15) .eq. 'wavelinearmulti') then
C              for wavelinearmulti, expect nbinc then 3*nbinc values
               if (param_count .ne. 0) then
                  call getarg(iargs+1, argval)
                  read(argval,*) nbinc
                  iargs = iargs + 1
                  ntemp = 4 * nbinc
                  if (param_count .eq. ntemp + 1) then
                     do 10 ipwave = 1, nbinc
                        call getarg(iargs + (ipwave-1)*4 + 1, argval)
                        read(argval,*) ppm(ipwave)
                        call getarg(iargs + (ipwave-1)*4 + 2, argval)
                        read(argval,*) ssm(ipwave)
                        call getarg(iargs + (ipwave-1)*4 + 3, argval)
                        read(argval, *) temp_real
                        call getarg(iargs + (ipwave-1)*4 + 4, argval)
                        read(argval, *) temp_imag
                        E0m(ipwave) = temp_real+(0.d0,1.d0)*temp_imag
10                   continue
                     iargs = iargs + ntemp
                  else
                     write(*,*) 'Error: Insufficient parameters'
                     stop
                  end if
               else
                  write(*,*) 'Error: requires nbinc and parameters'
                  stop
               end if
            else if (beam(1:7) .eq. 'antenna') then
C              for antenna, expect xgaus, ygaus, zgaus
               if (param_count .eq. 3) then
                  call getarg(iargs+1, argval)
                  read(argval,*) xgaus
                  call getarg(iargs+2, argval)
                  read(argval,*) ygaus
                  call getarg(iargs+3, argval)
                  read(argval,*) zgaus
                  iargs = iargs + 3
               else
                  write(*,*) 'Error: requires xgaus, ygaus, zgaus'
                  stop
               end if
            else if (beam(1:11) .eq. 'greentensor') then
C              for greentensor, expect xgaus, ygaus, zgaus
               if (param_count .eq. 3) then
                  call getarg(iargs+1, argval)
                  read(argval,*) xgaus
                  call getarg(iargs+2, argval)
                  read(argval,*) ygaus
                  call getarg(iargs+3, argval)
                  read(argval,*) zgaus
                  iargs = iargs + 3
               else
                  write(*,*) 'Error: requires xgaus, ygaus, zgaus'
                  stop
               end if
            else if (beam(1:13) .eq. 'pwavecircular') then
C              for pwavecircular, expect ss only
               if (param_count .eq. 1) then
                  call getarg(iargs+1, argval)
                  read(argval,*) ss
                  iargs = iargs + 1
               else
                  write(*,*) 'Error: requires ss'
                  stop
               end if
            else if (beam(1:11) .eq. 'gwavelinear') then
C              for gwavelinear, expect pp, ss, xgaus, ygaus, zgaus
               if (param_count .eq. 5) then
                  call getarg(iargs+1, argval)
                  read(argval,*) pp
                  call getarg(iargs+2, argval)
                  read(argval,*) ss
                  call getarg(iargs+3, argval)
                  read(argval,*) xgaus
                  call getarg(iargs+4, argval)
                  read(argval,*) ygaus
                  call getarg(iargs+5, argval)
                  read(argval,*) zgaus
                  iargs = iargs + 5
               else
                  write(*,*) 'Error: requires 5 parameters'
                  stop
               end if
            else if (beam(1:13) .eq. 'gwavecircular') then
C              for gwavecircular, expect ss, xgaus, ygaus, zgaus
               if (param_count .eq. 4) then
                  call getarg(iargs+1, argval)
                  read(argval,*) ss
                  call getarg(iargs+2, argval)
                  read(argval,*) xgaus
                  call getarg(iargs+3, argval)
                  read(argval,*) ygaus
                  call getarg(iargs+4, argval)
                  read(argval,*) zgaus
                  iargs = iargs + 4
               else
                  write(*,*) 'Error: requires 4 parameters'
                  stop
               end if
            else if (beam(1:15) .eq. 'gparawavelinear') then
C              for gparawavelinear, expect pp, ss, xgaus, ygaus, zgaus
               if (param_count .eq. 5) then
                  call getarg(iargs+1, argval)
                  read(argval,*) pp
                  call getarg(iargs+2, argval)
                  read(argval,*) ss
                  call getarg(iargs+3, argval)
                  read(argval,*) xgaus
                  call getarg(iargs+4, argval)
                  read(argval,*) ygaus
                  call getarg(iargs+5, argval)
                  read(argval,*) zgaus
                  iargs = iargs + 5
               else
                  write(*,*) 'Error: requires 5 parameters'
                  stop
               end if
            else if (beam(1:17) .eq. 'gparawavecircular') then
C              for gparawavecircular, expect ss, xgaus, ygaus, zgaus
               if (param_count .eq. 4) then
                  call getarg(iargs+1, argval)
                  read(argval,*) ss
                  call getarg(iargs+2, argval)
                  read(argval,*) xgaus
                  call getarg(iargs+3, argval)
                  read(argval,*) ygaus
                  call getarg(iargs+4, argval)
                  read(argval,*) zgaus
                  iargs = iargs + 4
               else
                  write(*,*) 'Error: requires ss, xgaus, ygaus, zgaus'
                  stop
               end if
            else if (beam(1:7) .eq. 'speckle') then
C              for speckle, expect speckseed, numaperil, xgaus, ygaus, zgaus
               if (param_count .eq. 5) then
                  call getarg(iargs+1, argval)
                  read(argval,*) speckseed
                  call getarg(iargs+2, argval)
                  read(argval,*) numaperil
                  call getarg(iargs+3, argval)
                  read(argval,*) xgaus
                  call getarg(iargs+4, argval)
                  read(argval,*) ygaus
                  call getarg(iargs+5, argval)
                  read(argval,*) zgaus
                  iargs = iargs + 5
               else
                  write(*,*) 'Errorr: requires 5 parameters'
                  stop
               end if
            else if (beam(1:11) .eq. 'demispeckle') then
C              for demispeckle, same as speckle
               if (param_count .eq. 5) then
                  call getarg(iargs+1, argval)
                  read(argval,*) speckseed
                  call getarg(iargs+2, argval)
                  read(argval,*) numaperil
                  call getarg(iargs+3, argval)
                  read(argval,*) xgaus
                  call getarg(iargs+4, argval)
                  read(argval,*) ygaus
                  call getarg(iargs+5, argval)
                  read(argval,*) zgaus
                  iargs = iargs + 5
               else
                  write(*,*) 'Error: requires 5 parameters'
                  stop
               end if
            else if (beam(1:8) .eq. 'confocal') then
C              for confocal, expect numaperil, xgaus, ygaus, zgaus
               if (param_count .eq. 4) then
                  call getarg(iargs+1, argval)
                  read(argval,*) numaperil
                  call getarg(iargs+2, argval)
                  read(argval,*) xgaus
                  call getarg(iargs+3, argval)
                  read(argval,*) ygaus
                  call getarg(iargs+4, argval)
                  read(argval,*) zgaus
                  iargs = iargs + 4
               else
                  write(*,*) 'Error: requires 4 parameters'
                  stop
               end if
            else if (beam(1:12) .eq. 'demiconfocal') then
C              for demiconfocal, expect numaperil, xgaus, ygaus, zgaus
               if (param_count .eq. 4) then
                  call getarg(iargs+1, argval)
                  read(argval,*) numaperil
                  call getarg(iargs+2, argval)
                  read(argval,*) xgaus
                  call getarg(iargs+3, argval)
                  read(argval,*) ygaus
                  call getarg(iargs+4, argval)
                  read(argval,*) zgaus
                  iargs = iargs + 4
               else
                  write(*,*) 'Error: requires 4 parameters'
                  stop
               end if
            else if (beam(1:9) .eq. 'arbitrary') then
C              for arbitrary, expect a filename for namefileinc
               if (param_count .eq. 1) then
                  call getarg(iargs+1, argval)
                  namefileinc = trim(argval)
                  iargs = iargs + 1
               else
                  write(*,*) 'Error: requires a filename'
                  stop
               end if
            else
               write(*,*) 'Error: unrecognized beam type:', beam
               stop
            end if


         else if (trim(arg) .eq. '-prop') then
C           Process the -prop option:
C           if nbinc = 1, expect 2 parameters for theta and phi.
C           if nbinc > 1, expect 2*nbinc parameters for thetam and phim.
            param_count = GET_PARAM_COUNT(iargs+1, nargs)

            if (nbinc .eq. 1) then
               if (param_count .eq. 2) then
                  call getarg(iargs+1, argval)
                  read(argval,*) theta
                  call getarg(iargs+2, argval)
                  read(argval,*) phi
                  iargs = iargs + 2
               else
                  write(*,*) 'Error: requires theta and phi'
                  stop
               endif
            else
               if (param_count .eq. 2*nbinc) then
                  do 20 iprop = 1, nbinc
                     call getarg(iargs + 2*(iprop-1) + 1, argval)
                     read(argval,*) thetam(iprop)
20                continue
                  do 30 iprop = 1, nbinc
                     call getarg(iargs + 2*(iprop-1) + 2, argval)
                     read(argval,*) phim(iprop)
30                continue
                  iargs = iargs + 2*nbinc
               else
                  write(*,*) 'Error: requires theta and phi * nbinc'
                  stop
               endif
            endif

C        object type; if missing, default to sphere
         else if (trim(arg) .eq. '-object') then
            iargs = iargs + 1
            if (iargs .gt. nargs) then
               write(*,*) 'Error: -object without type'
               stop
            end if
            call getarg(iargs, object)

            param_count = GET_PARAM_COUNT(iargs+1, nargs)

            if (len_trim(object) .eq. 6 .and. object(1:6) .eq.
     $         'sphere') then
               if (param_count .eq. 1) then
                  call getarg(iargs+1, argval)
                  read(argval, *) rayonmulti(1)
                  iargs = iargs + 1
               else if (param_count .eq. 0) then
                  write(*,*) 'Using default values for sphere'
               else
                  write(*,*) 'Error: requires 0 or 1 parameters'
                  stop
               endif
            else if (object(1:12) .eq. 'inhomosphere') then
               if (param_count .eq. 4) then
                  call getarg(iargs+1, argval)
                  read(argval, *) rayonmulti(1)
                  call getarg(iargs+2, argval)
                  read(argval, *) lc
                  call getarg(iargs+3, argval)
                  read(argval, *) hc
                  call getarg(iargs+4, argval)
                  read(argval, *) ng
                  iargs = iargs + 4
               else
                  write(*,*) 'Error: requires 4 parameters'
                  stop
               endif
            else if (object(1:13) .eq. 'inhomocuboid1') then
               if (param_count .eq. 6) then
                  call getarg(iargs+1, argval)
                  read(argval, *) sidex
                  call getarg(iargs+2, argval)
                  read(argval, *) sidey
                  call getarg(iargs+3, argval)
                  read(argval, *) sidez
                  call getarg(iargs+4, argval)
                  read(argval, *) lc
                  call getarg(iargs+5, argval)
                  read(argval, *) hc
                  call getarg(iargs+6, argval)
                  read(argval, *) ng
                  iargs = iargs + 6
               else
                  write(*,*) 'Error: requires or  6 parameters'
                  stop
               endif
            else if (object(1:13) .eq. 'inhomocuboid2') then
               if (param_count .eq. 4) then
                  call getarg(iargs+1, argval)
                  read(argval, *) aretecube
                  call getarg(iargs+2, argval)
                  read(argval, *) lc
                  call getarg(iargs+3, argval)
                  read(argval, *) hc
                  call getarg(iargs+4, argval)
                  read(argval, *) ng
                  iargs = iargs + 4
               else
                  write(*,*) 'Error: requires or 4 parameters'
                  stop
               endif
            else if (object(1:13) .eq. 'randomsphere1') then
               if (param_count .eq. 6) then
                  call getarg(iargs+1, argval)
                  read(argval, *) rayonmulti(1)
                  call getarg(iargs+2, argval)
                  read(argval, *) density
                  call getarg(iargs+3, argval)
                  read(argval, *) sidex
                  call getarg(iargs+4, argval)
                  read(argval, *) sidey
                  call getarg(iargs+5, argval)
                  read(argval, *) sidez
                  call getarg(iargs+6, argval)
                  read(argval, *) ng
                  iargs = iargs + 6
               else
                  write(*,*) 'Error: requires 6 parameters'
                  stop
               endif
            else if (object(1:13) .eq. 'randomsphere2') then
               if (param_count .eq. 4) then
                  call getarg(iargs+1, argval)
                  read(argval, *) rayonmulti(1)
                  call getarg(iargs+2, argval)
                  read(argval, *) density
                  call getarg(iargs+3, argval)
                  read(argval, *) aretecube
                  call getarg(iargs+4, argval)
                  read(argval, *) ng
                  iargs = iargs + 4
               else
                  write(*,*) 'Error: requires 4 parameters or none'
                  stop
               endif
            else if (object(1:4) .eq. 'cube') then
               if (param_count .eq. 1) then
                  call getarg(iargs+1, argval)
                  read(argval, *) side
                  iargs = iargs + 1
               else
                  write(*,*) 'Error: requires 1 parameter'
                  stop
               endif
            else if (object(1:7) .eq. 'cuboid1') then
               if (param_count .eq. 3) then
                  call getarg(iargs+1, argval)
                  read(argval, *) sidex
                  call getarg(iargs+2, argval)
                  read(argval, *) sidey
                  call getarg(iargs+3, argval)
                  read(argval, *) sidez
                  iargs = iargs + 3
               else
                  write(*,*) 'Error: requires 3 parameters'
                  stop
               endif
            else if (object(1:7) .eq. 'cuboid2') then
               if (param_count .eq. 1) then
                  call getarg(iargs+1, argval)
                  read(argval, *) aretecube
                  iargs = iargs + 1
               else
                  write(*,*) 'Error: requires 1 parameter'
                  stop
               endif
            else if (object(1:8) .eq. 'nspheres') then
               call getarg(iargs+1, argval)
               read(argval, *) numberobjet
               if (param_count .eq. numberobjet + 1) then
                  do 40 iobj = 1, numberobjet
                     call getarg(iargs+1+iobj, argval)
                     read(argval, *) rayonmulti(iobj)
40                   continue
                  iargs = iargs + numberobjet + 1
               else
                  write(*,*) 'Error: requires nobj+1 parameters'
                  stop
               endif
            else if (object(1:9) .eq. 'ellipsoid') then
               if (param_count .eq. 3) then
                  call getarg(iargs+1, argval)
                  read(argval, *) demiaxea
                  call getarg(iargs+2, argval)
                  read(argval, *) demiaxeb
                  call getarg(iargs+3, argval)
                  read(argval, *) demiaxec
                  iargs = iargs + 3
               else
                  write(*,*) 'Error: requires 3 parameters'
                  stop
               endif
            else if (object(1:8) .eq. 'cylinder') then
               if (param_count .eq. 2) then
                  call getarg(iargs+1, argval)
                  read(argval, *) rayonmulti(1)
                  call getarg(iargs+2, argval)
                  read(argval, *) hauteur
                  iargs = iargs + 2
               else
                  write(*,*) 'Error: requires 2 parameters'
                  stop
               endif
            else if (len_trim(object) .eq. 16 .and. object(1:16) .eq.
     $         'concentricsphere') then
               call getarg(iargs+1, argval)
               read(argval, *) numberobjet
               if (param_count .eq. numberobjet + 1) then
                  do 50 iobj = 1, numberobjet
                     call getarg(iargs+1+iobj, argval)
                     read(argval, *) rayonmulti(iobj)
50                   continue
                  iargs = iargs + numberobjet + 1
               else
                  write(*,*) 'Error: requires nobj+1 parameters'
                  stop
               endif
            else if (object(1:9) .eq. 'arbitrary') then
               if (param_count .eq. 1) then
                  call getarg(iargs+1, argval)
                  namefileobj = trim(argval)
                  iargs = iargs + 1
               else
                  write(*,*) 'Error: requires 1 parameter'
                  stop
               end if

               open(15, file=namefileobj, status='old', iostat=ierror)
               if (ierror .ne. 0) then
                  write(*,*) 'bad namefile for arbitrary'
                  stop
               end if
               read(15,*) nx, ny, nz
               read(15,*) aretecube
               rewind(15)
               close(15)

               if (nx .gt. nxm .or. ny .gt. nym .or. nz .gt. nzm) then
                  write(*,*) 'Size of the table too small'
                  stop
               end if
            else
               write(*,*) 'Error: unrecognized object type: ', object
               stop
            end if
         else if (trim(arg) .eq. "-orient") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 3) then
               call getarg(iargs+1, argval)
               read(argval,*) phiobj
               call getarg(iargs+2, argval)
               read(argval,*) thetaobj
               call getarg(iargs+3, argval)
               read(argval,*) psiobj
               iargs = iargs + 3
            else
               write(*,*) 'Error: requires phiobj, thetaob, psiobj'
               stop
            endif
         else if (trim(arg) .eq. '-pos') then
C           Process the -pos option:
C           if numberobjet = 1, expect 3 parameters for xgmulti ygmulti zgmulti.
C           if numberobjet > 1, expect 3*numberobjet parameters.
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (numberobjet .eq. 1) then
               if (param_count .eq. 3) then
                  call getarg(iargs+1, argval)
                  read(argval,*) xgmulti(1)
                  call getarg(iargs+2, argval)
                  read(argval,*) ygmulti(1)
                  call getarg(iargs+3, argval)
                  read(argval,*) zgmulti(1)
                  iargs = iargs + 3
               else
                  write(*,*) 'Error: requires 3 parameters'
                  stop
               endif
            else
               if (param_count .eq. 3 * numberobjet) then
                  do 60 ipos = 1, numberobjet
                     call getarg(iargs + 3*(ipos-1) + 1, argval)
                     read(argval, *) xgmulti(ipos)
                     call getarg(iargs + 3*(ipos-1) + 2, argval)
                     read(argval, *) ygmulti(ipos)
                     call getarg(iargs + 3*(ipos-1) + 3, argval)
                     read(argval, *) zgmulti(ipos)
60                   continue
                  iargs = iargs + 3 * numberobjet
               else
                  write(*,*) 'Error: requires 3*numberobjet parameters'
                  stop
               endif
            endif

         else if (trim(arg) == "-nnnr") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               iargs = iargs + 1
               call getarg(iargs, argval)
               read(argval, *) nnnr
               nxm = nnnr
               nym = nnnr
               nzm = nnnr
            else if (param_count .eq. 4) then
               call getarg(iargs+1, argval)
               read(argval, *) nnnr
               call getarg(iargs+2, argval)
               read(argval, *) nxm
               call getarg(iargs+3, argval)
               read(argval, *) nym
               call getarg(iargs+4, argval)
               read(argval, *) nzm
               iargs = iargs + 4
            else
               write(*,*) 'Error: requires 1 or 4 parameters'
               stop
            endif
         else if (trim(arg) == "-polarizability") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               iargs = iargs + 1
               call getarg(iargs, argval)
               polarizability = trim(adjustl(argval))  ! Handling string argument
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-methodeit") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               iargs = iargs + 1
               call getarg(iargs, argval)
               methodeit = trim(adjustl(argval))  ! Handling string argument
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-tolinit") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               iargs = iargs + 1
               call getarg(iargs, argval)
               read(argval, *) tolinit
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-ninitest") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               iargs = iargs + 1
               call getarg(iargs, argval)
               read(argval, *) ninitest
            else if (param_count .eq. 3) then
               call getarg(iargs+1, argval)
               read(argval, *) ninitest
               call getarg(iargs+2, argval)
               read(argval, *) nlecture
               call getarg(iargs+3, argval)
               filereread = trim(adjustl(argval))  ! Handling string argument
               iargs = iargs + 3
            else
               write(*,*) "Error: requires 1 or 3 parameters"
               stop
            endif
         else if (trim(arg) == "-nrig") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               iargs = iargs + 1
               call getarg(iargs, argval)
               read(argval, *) nrig
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-nlar") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               iargs = iargs + 1
               call getarg(iargs, argval)
               read(argval, *) nlar
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-nlim") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               iargs = iargs + 1
               call getarg(iargs, argval)
               read(argval, *) nlim
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-nprecon") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               iargs = iargs + 1
               call getarg(iargs, argval)
               read(argval, *) nprecon
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-nforce") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 4) then
               call getarg(iargs+1, argval)
               read(argval, *) nforce
               call getarg(iargs+2, argval)
               read(argval, *) nforced
               call getarg(iargs+3, argval)
               read(argval, *) ntorque
               call getarg(iargs+4, argval)
               read(argval, *) ntorqued
               iargs = iargs + 4
            else
               write(*,*) "Error: requires 4 parameters"
               stop
            endif
         else if (trim(arg) == "-cross_section") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               call getarg(iargs+1, argval)
               read(argval, *) nsection
               iargs = iargs + 1
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-energy") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               call getarg(iargs+1, argval)
               read(argval, *) nenergie
               iargs = iargs + 1
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-Csca") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               call getarg(iargs+1, argval)
               read(argval, *) nsectionsca
               iargs = iargs + 1
            else if (param_count .eq. 3) then
               call getarg(iargs+1, argval)
               read(argval, *) nsectionsca
               call getarg(iargs+2, argval)
               read(argval, *) ntheta
               call getarg(iargs+3, argval)
               read(argval, *) nphi
               iargs = iargs + 3
            else
               write(*,*) "Error: requires 1 or 3 parameters"
               stop
            endif
         else if (trim(arg) == "-quick_FFT") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               call getarg(iargs+1, argval)
               read(argval, *) nquickdiffracte
               iargs = iargs + 1
            else if (param_count .eq. 2) then
               call getarg(iargs+1, argval)
               read(argval, *) nquickdiffracte
               call getarg(iargs+2, argval)
               read(argval, *) nfft2d
               iargs = iargs + 2
            else
               write(*,*) "Error: requires 1 or 2 parameters"
               stop
            endif
         else if (trim(arg) == "-near_field") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 3) then
               call getarg(iargs+1, argval)
               read(argval, *) nlocal
               call getarg(iargs+2, argval)
               read(argval, *) nmacro
               call getarg(iargs+3, argval)
               read(argval, *) nproche
               iargs = iargs + 3
            else if (param_count .eq. 6) then
               call getarg(iargs+1, argval)
               read(argval, *) nlocal
               call getarg(iargs+2, argval)
               read(argval, *) nmacro
               call getarg(iargs+3, argval)
               read(argval, *) nproche
               call getarg(iargs+4, argval)
               read(argval, *) nxmp
               call getarg(iargs+5, argval)
               read(argval, *) nymp
               call getarg(iargs+6, argval)
               read(argval, *) nzmp
               iargs = iargs + 6
            else
               write(*,*) "Error: requires 3 or 6 parameters"
               stop
            endif
         else if (trim(arg) == "-only_dipoles") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               call getarg(iargs+1, argval)
               read(argval, *) nobjet
               iargs = iargs + 1
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-igt") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               call getarg(iargs+1, argval)
               read(argval, *) nquad
               iargs = iargs + 1
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-lambda") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               call getarg(iargs+1, argval)
               read(argval, *) lambda
               iargs = iargs + 1
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-power") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               call getarg(iargs+1, argval)
               read(argval, *) P0
               iargs = iargs + 1
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-waist") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               call getarg(iargs+1, argval)
               read(argval, *) w0
               iargs = iargs + 1
            else
               write(*,*) "Error: requires 1 parameters"
               stop
            endif
         else if (trim(arg) == "-save_data") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 1) then
               iargs = iargs + 1
               call getarg(iargs, argval)
               read(argval, *) nmat
            else if (param_count .eq. 2) then
               call getarg(iargs+1, argval)
               read(argval, *) nmat
               call getarg(iargs+2, argval)
               h5file = trim(adjustl(argval))  ! Handling string argument
               iargs = iargs + 2
            else
               write(*,*) "Error: requires 1 or 2 parameters"
               stop
            endif
         else if (trim(arg) == "-epsmulti") then
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (param_count .eq. 2*numberobjet) then
               do 70 ieps = 1, numberobjet
                  call getarg(iargs + 2*(ieps-1) + 1, argval)
                  read(argval, *) epsreal
                  call getarg(iargs + 2*(ieps-1) + 2, argval)
                  read(argval, *) epsimag
!     epsmulti(ieps) = cmplx(epsreal, epsimag)
                  epsmulti(ieps) = epsreal+(0.d0,1.d0)* epsimag
70             continue
               iargs = iargs + 2 * numberobjet
            else
               write(*,*) "Error: requires Re{eps},Im{eps}*numberobject"
               stop
            endif
         else if (trim(arg) .eq. '-h' .or. trim(arg) .eq. '--help') then
            write(*,*) 'Usage: ./ifdda [options]'
            write(*,*) ' -beam <type> [params] :'
            write(*,*) '      Beam type & parameters. Supported:'
            write(*,*) '       pwavelinear, wavelinearmulti, antenna,'
            write(*,*) '       greentensor, pwavecircular, gwavelinear,'
            write(*,*) '            gwavecircular, gparawavelinear,'
            write(*,*) '       gparawavecircular, speckle, demispeckle,'
            write(*,*) '       confocal, demiconfocal, arbitrary'
            write(*,*) ' -methodeit <string> :'
            write(*,*) '      Iterative method. Supported:'
            write(*,*) '       GPBICG2, GPBICGplus, GPBICGsafe,'
            write(*,*) '       GPBICGAR, GPBICGAR2, BICGSTARPLUS,'
            write(*,*) '       GPBICOR, CORS, QMRCLA, TFQMR, CG,'
            write(*,*) '       BICGSTAB, QMRBICGSTAB1, QMRBICGSTAB2,'
            write(*,*) '       IDRS2, IDRS4, IDRS8, BICGLSTABL2,'
            write(*,*) '       BICGLSTABL4, BICGLSTABL8,'
            write(*,*) '       GPBICGSTABL2, GPBICGSTABL4, GPBICGSTABL8'
            write(*,*) ' -object <type> [params] :'
            write(*,*) '      Object type & parameters. Supported:'
            write(*,*) '       sphere, inhomosphere,'
            write(*,*) '       inhomocuboid1, inhomocuboid2,'
            write(*,*) '       randomsphere1, randomsphere2, cube,'
            write(*,*) '       cuboid1, cuboid2, nspheres, ellipsoid,'
            write(*,*) '       cylinder, sphereconcentric, arbitrary'
            write(*,*) ' -polarizability <string> :'
            write(*,*) '      Supported: CM, RR, LA, LR, GB, PS, FG'
            write(*,*) ' -prop <...> : Propagation parameters:'
            write(*,*) '      if nbinc=1, provide theta and phi;'
            write(*,*) '      if nbinc>1, provide 2*nbinc values.'
            write(*,*) ' -orient <phi> <theta> <psi> :'
            write(*,*) '      Object orientation angles.'
            write(*,*) ' -pos <x> <y> <z> :'
            write(*,*) '      Object position (x,y,z per object).'
            write(*,*) ' -nnnr <D> or <D> <nx> <ny> <nz> :'
            write(*,*) '      Grid discretization.'
            write(*,*) ' -tolinit <value> : e.g. 1.d-4'
            write(*,*) ' -ninitest <v> or <v> <nlect> <file> :'
            write(*,*) '      Initial guess option.'
            write(*,*) ' -nrig <value> : nrig option.'
            write(*,*) ' -nlar <value> : nlar option.'
            write(*,*) ' -nlim <value> : Maximum iterations.'
            write(*,*) ' -nprecon <value> : Preconditioner type.'
            write(*,*) ' -nforce <v> <v> <v> <v> : Force parameters.'
            write(*,*) ' -cross_section <value> : Cross section.'
            write(*,*) ' -energy <value> : Energy parameter.'
            write(*,*) ' -Csca <v> or <v> <nt> <np> : Csca parameters.'
            write(*,*) ' -quick_FFT <v> or <v> <nfft2d> : Quick FFT.'
            write(*,*) ' -near_field <nloc> <nmac> <nprox>'
            write(*,*) '      Optional [<nx> <ny> <nz>].'
            write(*,*) '      Near field parameters.'
            write(*,*) ' -only_dipoles <value> : Only dipoles option.'
            write(*,*) ' -igt <value> : IGT option.'
            write(*,*) ' -lambda <value> : Wavelength (nm).'
            write(*,*) ' -power <value> : Power.'
            write(*,*) ' -waist <value> : Waist (nm).'
            write(*,*) ' -save_data <v> or <v> <h5file> : Save data.'
            write(*,*) ' -epsmulti <r> <r> : Complex permittivity.'
            write(*,*) ' -trope [params] : Trope option.'
            write(*,*) ' -h, --help : Print this help message'
            stop
         else if (trim(arg) == "-trope") then
            trope = 'ani'
            param_count = GET_PARAM_COUNT(iargs+1, nargs)
            if (numberobjet .eq. 1) then
               if (param_count .eq. 0) then
                  write(*,*) "Using default values for trope"
                  epsanimulti(1,1,1) = (2.D0, 0.D0)
                  epsanimulti(2,1,1) = (0.D0, 0.D0)
                  epsanimulti(3,1,1) = (0.D0, 0.D0)
                  epsanimulti(1,2,1) = (0.D0, 0.D0)
                  epsanimulti(2,2,1) = (2.D0, 0.D0)
                  epsanimulti(3,2,1) = (0.D0, 0.D0)
                  epsanimulti(1,3,1) = (0.D0, 0.D0)
                  epsanimulti(2,3,1) = (0.D0, 0.D0)
                  epsanimulti(3,3,1) = (2.D0, 0.D0)
               else if (param_count .eq. 18) then
                  itoken = 0
                  do 90 icol = 1, 3
                     do 80 iraw = 1, 3
                        itoken = itoken + 1
                        call getarg(iargs + 2*itoken - 1, argval)
                        read(argval, *) epsanimultireal
                        call getarg(iargs + 2*itoken, argval)
                        read(argval, *) epsanimultiimag
                        epsanimulti(iraw, icol, 1) =epsanimultireal
     $                       +(0.d0,1.d0) *epsanimultiimag
80                   continue
90                continue
                  iargs = iargs + 18
               else
                  write(*,*) "Error: requires 18 parameters"
                  stop
               endif
            else
               if (param_count .eq. 18*numberobjet) then
                  itoken = 0
                  do 130 ieps = 1, numberobjet
                     do 120 icol = 1, 3
                        do 110 iraw = 1, 3
                           itoken = itoken + 1
                           call getarg(iargs + 2*itoken - 1, argval)
                           read(argval, *) epsanimultireal
                           call getarg(iargs + 2*itoken, argval)
                           read(argval, *) epsanimultiimag
                           epsanimulti(iraw, icol, ieps) =
     $                          epsanimultireal +(0.d0,1.d0)
     $                          *epsanimultiimag
110                     continue
120                  continue
130               continue
                  iargs = iargs + 18*numberobjet
               else
                  write(*,*) "Error: requires 18*numberobjet parameters"
                  stop
               endif
            endif

         else
            write(*,*) 'Error: unrecognized argument type: ', arg
            stop
         end if

         iargs = iargs + 1
         go to 100
200   continue

      alloc_size = max((ntheta+1)*nphi, nfft2d * nfft2d)
      allocate(thetafield(alloc_size))
      allocate(phifield(alloc_size))
      allocate(poyntingfield(alloc_size))
      allocate(poyntingfieldpos(alloc_size))
      allocate(poyntingfieldneg(alloc_size))
      allocate(xs(nxm*nym*nzm), ys(nxm*nym*nzm), zs(nxm*nym*nzm))
      allocate(xswf(nxm*nym*nzm), yswf(nxm*nym*nzm), zswf(nxm*nym*nzm))
      allocate(polarisa(nxm*nym*nzm,3,3), epsilon(nxm*nym*nzm,3,3))
      allocate(incidentfield(nxm*nym*nzm), localfield(nxm*nym*nzm))
      allocate(macroscopicfield(nxm*nym*nzm))
      allocate(forcex(nxm*nym*nzm), forcey(nxm*nym*nzm))
      allocate(forcez(nxm*nym*nzm))
      allocate(torquex(nxm*nym*nzm),torquey(nxm*nym*nzm))
      allocate(torquez(nxm*nym*nzm))
      allocate(macroscopicfieldx(nxm*nym*nzm))
      allocate(macroscopicfieldy(nxm*nym*nzm))
      allocate(macroscopicfieldz(nxm*nym*nzm))
      allocate(localfieldx(nxm*nym*nzm), localfieldy(nxm*nym*nzm))
      allocate(localfieldz(nxm*nym*nzm))
      allocate(incidentfieldx(nxm*nym*nzm), incidentfieldy(nxm*nym*nzm))
      allocate(incidentfieldz(nxm*nym*nzm))
      allocate(FF(3*nxm*nym*nzm), FF0(3*nxm*nym*nzm))
      allocate(FFloc(3*nxm*nym*nzm))
      allocate(Tabdip(nxm*nym*nzm), Tabmulti(nxm*nym*nzm))
      allocate(FFTTENSORxx(8*nxm*nym*nzm), FFTTENSORxy(8*nxm*nym*nzm))
      allocate(FFTTENSORxz(8*nxm*nym*nzm))
      allocate(FFTTENSORyy(8*nxm*nym*nzm), FFTTENSORyz(8*nxm*nym*nzm))
      allocate(FFTTENSORzz(8*nxm*nym*nzm))
      allocate(vectx(8*nxm*nym*nzm),vecty(8*nxm*nym*nzm))
      allocate(vectz(8*nxm*nym*nzm))
      allocate(xr(3*nxm*nym*nzm), xi(3*nxm*nym*nzm))
      allocate(wrk(3*nxm*nym*nzm, nlar))
      allocate(tabfft2(nfft2d))
      allocate(kxy(nfft2d), xy(nfft2d),kxypoynting(nfft2d))
      allocate(Eimagex(nfft2d*nfft2d),Eimagey(nfft2d*nfft2d))
      allocate(Eimagez(nfft2d*nfft2d),Eimageincx(nfft2d*nfft2d))
      allocate(Eimageincy(nfft2d *nfft2d) ,Eimageincz(nfft2d*nfft2d))
      allocate(Efourierx(nfft2d*nfft2d) ,Efouriery(nfft2d*nfft2d))
      allocate(Efourierz(nfft2d*nfft2d),Efourierincx(nfft2d*nfft2d))
      allocate(Efourierincy(nfft2d*nfft2d) ,Efourierincz(nfft2d*nfft2d))
      allocate(Ediffkzpos(nfft2d,nfft2d,3),Ediffkzneg(nfft2d ,nfft2d,3))

      if (numberobjet.gt.numberobjetmax) then
         write(*,*) 'redimensionner numberobjetmax',numberobjet
     $        ,numberobjetmax
         stop
      endif

      if (nprecon.eq.0) then
         nxym1=1
         nzm1=1
         nzms1=1
      else
         if (nrig.eq.0) then
            nxym1=nxm*nym
            nzm1=nzm
            nzms1=3*nzm
         else
            nxym1=nxm*nym
            nzm1=nzm
            nzms1=nzm
         endif
      endif
      allocate(sdetnn(nzms1,nzms1,nxym1))
      allocate(FFprecon(3*nxym1*nzm1))


c********************************************************
c     Microscopy (TODO)
c********************************************************

      nside=0                   ! compute microscope in reflexion (1), or transmission (0).
      nlentille=0               ! Compute microscopy.
      nquicklens=0              ! Compute microscopy with FFT (1) or wihtout FFT (0).
      numaper= 0.9d0            ! Numerical aperture for the microscope.
      zlens=0.d0              ! Position of lens.
      ntypemic=0                ! Type of microscope: O Holographic, 1 Bright field, 2 Dark field, 3 schiren, 4 Dark field2, 5 dark field experimental, 6 confocal
      gross=100.d0              ! Manyfing factor for the microscope
      numaperinc=0.9d0          ! Numerical aperture for the condenser lens.
      numaperinc2=0.0d0         ! Numerical aperture for central aperture
      kcnax=0.d0                ! X position of the NA
      kcnay=0.d0                ! Y position of the NA


c*******************************************************
c     Confocal microscope
c*******************************************************
      if (ntypemic.eq.6) then
         zstep=100.d0            !step of the z scan to create the z-stack
         zinf=-1000.d0          !lower z of the s-stack
         zsup=1000.d0           !higher z of the s-stack

c     use previous varaibles
         zlens=zstep
         kcnax=zinf
         kcnay=zsup
      endif
c******************************************************

c     compute size when meshsize is given
      if (object(1:13).eq.'inhomocuboid2'.or.object(1:7).eq.'cuboid2')
     $     then
         nx=nxm-2*nxmp
         ny=nym-2*nymp
         nz=nzm-2*nzmp
      else
         nx=nnnr
         ny=nnnr
         nz=nnnr
         if (nx+2*nxmp.gt.nxm) then
            write(*,*) 'pb with size: increase nxm'
            stop
         endif
         if (ny+2*nymp.gt.nym) then
            write(*,*) 'pb with size: increase nym'
            stop
         endif
         if (nz+2*nzmp.gt.nzm) then
            write(*,*) 'pb with size: increase nzm'
            stop
         endif
      endif


      call cdmlib(
c     input file cdm.in
     $     lambda,beam,object,trope,
     $     materiaumulti,nnnr,tolinit,nlim,methodeit,polarizability
     $     ,nprecon,ninitest,nquad,nlecture,filereread,nmat,h5file,
c     output file cdm.out
     $     nlocal,nmacro,nsection,nsectionsca,nquickdiffracte,nrig,
     $     nforce,nforced,ntorque,ntorqued,nproche,nlentille,nquicklens,
     $     nenergie,nobjet,
c     cube, sphere (includes multiple)
     $     density,side, sidex, sidey, sidez, hauteur,
     $     numberobjet, rayonmulti, xgmulti, ygmulti, zgmulti,
     $     epsmulti, epsanimulti,lc,hc,ng,
c     ellipsoid+arbitrary
     $     demiaxea,demiaxeb,demiaxec,thetaobj,phiobj,psiobj,
     $     namefileobj,
c     planewavecircular.in / planewavelinear.in files
     $     theta, phi, pp, ss, P0, w0, xgaus, ygaus, zgaus,speckseed,
     $     namefileinc,
c     ondeplane multiple
     $     thetam, phim, ppm, ssm,E0m,nbinc,
c     return info stringf
     $     infostr, nstop,
c     return scalar results
     $     nbsphere, ndipole, aretecube,
     $     lambda10n,k0,tol1,tempsreelmvp,tempsreeltotal,ncompte,nloop,
     $     efficacite,efficaciteref,efficacitetrans,
     $     Cext, Cabs, Csca, Cscai, gasym, irra, E0,
     $     forcet, forcem,
     $     couplet, couplem,
     $     nxm, nym, nzm, nxmp, nymp, nzmp, nmaxpp,nxym1, nzm1, nzms1,
     $     incidentfield, localfield, macroscopicfield,
     $     xs, ys, zs, xswf, yswf, zswf,
     $     ntheta, nphi, thetafield, phifield, poyntingfield,
     $     kxypoynting,poyntingfieldpos, poyntingfieldneg,
     $     forcex,forcey,forcez,forcexmulti,forceymulti,forcezmulti,
     $     torquex,torquey,torquez,torquexmulti,torqueymulti,
     $     torquezmulti,
     $     incidentfieldx, incidentfieldy, incidentfieldz,
     $     localfieldx, localfieldy, localfieldz,
     $     macroscopicfieldx, macroscopicfieldy, macroscopicfieldz,
     $     polarisa,epsilon,
     $     nfft2d,npoynting,deltapoyntingx,deltapoyntingy,
     $     Eimagex,Eimagey,Eimagez,Eimageincx,Eimageincy,
     $     Eimageincz,Efourierx,Efouriery,Efourierz,Efourierincx,
     $     Efourierincy,Efourierincz,kxy,xy,numaper,numaperinc
     $     ,numaperinc2,kcnax,kcnay,numaperil,gross,zlens,psiinc
     $     ,ntypemic,nside,
c****************************************************
c     tableaux utilises que dans cdmlib
c****************************************************
c     taille double complex (3*nxm*nym*nzm)
     $     FF,FF0,FFloc,xr,xi,FFprecon,
c     taille double complex (3*nxm*nym*nzm,12)
     $     wrk,nlar,
c     taille double complex (8*nxm*nym*nzm)
     $     FFTTENSORxx, FFTTENSORxy,FFTTENSORxz,FFTTENSORyy,FFTTENSORyz,
     $     FFTTENSORzz,vectx,vecty,vectz,
c     taille double complex (nfft2d,nfft2d,3)
     $     Ediffkzpos,Ediffkzneg,sdetnn,
c     taille entier (nxm*nym*nzm)
     $     Tabdip,Tabmulti,tabfft2)
c     output
      deallocate(sdetnn)
      deallocate(FFprecon)
      deallocate(thetafield,phifield)
      deallocate(poyntingfield,poyntingfieldneg,poyntingfieldpos)
      deallocate(xs, ys, zs, xswf, yswf)
      deallocate(polarisa, epsilon)
      deallocate(incidentfield, localfield, macroscopicfield)
      deallocate(forcex, forcey, forcez)
      deallocate(torquex, torquey, torquez)
      deallocate(macroscopicfieldx, macroscopicfieldy)
      deallocate(macroscopicfieldz)
      deallocate(localfieldx, localfieldy, localfieldz)
      deallocate(incidentfieldx, incidentfieldy, incidentfieldz)
      deallocate(FF, FF0, FFloc)
      deallocate(Tabdip, Tabmulti)
      deallocate(FFTTENSORxx, FFTTENSORxy, FFTTENSORxz)
      deallocate(FFTTENSORyy, FFTTENSORyz, FFTTENSORzz)
      deallocate(vectx, vecty, vectz)
      deallocate(xr, xi)
      deallocate(wrk)
      deallocate(tabfft2)
      deallocate(kxy,xy,kxypoynting)
      deallocate(Eimagex,Eimagey)
      deallocate(Eimagez,Eimageincx)
      deallocate(Eimageincy,Eimageincz)
      deallocate(Efourierx,Efouriery)
      deallocate(Efourierz,Efourierincx)
      deallocate(Efourierincy,Efourierincz)
      deallocate(Ediffkzpos,Ediffkzneg)


      if (nstop.eq.0) then
         write(*,*) '***********************************************'
         write(*,*) 'Computation finished without problem:'
         write(*,*) '***********************************************'
      else
         write(*,*) '***********************************************'
         write(*,*) 'Programm finished with problem:'
         write(*,*) infostr
         write(*,*) '***********************************************'
      endif


      end

c********************************************************
c     !Function counting the number of params per command-line argument
c********************************************************

      integer function GET_PARAM_COUNT(start_index, nargs)
      implicit none
      integer start_index, nargs, j, count, ios
      character(len=100) argval
      real dummy

      count = 0
      do j = start_index, nargs
         call getarg(j, argval)

         if (argval(1:1) == '-') then
c           Check if it's a negative number or an argument
            read(argval, *, iostat=ios) dummy
            if (ios /= 0) exit  ! not a number → it's a flag
         end if

         count = count + 1
      end do

      GET_PARAM_COUNT = count
      return
      end function
