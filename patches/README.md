# DDSCAT vs ADDA Dipole Polarizability Comparison

## Purpose

The modifications introduced in `DDPOSTPROCESS.f90` add a new DDSCAT post-processing output file:

- `ddpostprocess_P3D.out`

This file is used to compare DDSCAT dipole polarizabilities with ADDA dipole polarizabilities stored in:

- `DipPol`

## Example command lines

```bash
ddscatcli -CMDSOL QMRCCG -CALPHA FLTRCD -MEM_ALLOW "15 15 15" -CSHAPE ELLIPSOID -SHPAR "15 15 15" -DIEL "diel/m1.302265942870714_0.07678923076154483" -WAVELENGTHS "0.5 0.5 1 'LIN'" -AEFF "0.1 0.1 1 'LIN'" -TOL 1.e-4 -IORTH 1 -POL_E01 "(0,0) (1,0) (0,0)" -ROT_BETA "0 0 1" -ROT_THETA "0 0 1" -ROT_PHI "0 0 1" -CMDTRQ NOTORQ -ETASCA 10 -NRFLD 1 -run -post
./adda -lambda 0.5 -shape sphere -size 0.2 -iter qmr -m 1.302265942870714 0.07678923076154483 -eps 4 -pol fcd -int fcd -grid 15 -init_field zero -scat dr -prop 1 0 0 -store_dip_pol
```

## Example first line comparison

### DDSCAT

```text
# x y z  Re(Px) Im(Px) Re(Py) Im(Py) Re(Pz) Im(Pz)
    -1.3273873499531198E-02   -2.6547746999062396E-02   -9.2917114496718384E-02    -1.8228917754098991E-09    6.6880932730505111E-09    1.0102454628044824E-07    1.5947894197500357E-08    4.5141043562858829E-09    3.7515901184663285E-09
```

### ADDA

```text
x y z |P|^2 Px.r Px.i Py.r Py.i Pz.r Pz.i
-0.01327387349953119 -0.02654774699906239 -0.09291711449671836 1.054279937311537e-14 -1.822891775409916e-09 6.688093273050538e-09 1.010245462804481e-07 1.594789419750035e-08 4.51410435628589e-09 3.751590118466345e-09
```