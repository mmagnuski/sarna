# turning EasyCap channel locs to EEGLAB-Matlab spherical locs:

Matlab Spherical
----------------
`sph_theta` [real degrees] Matlab spherical horizontal angle.
            positive => rotating from nose (0) toward left ear.
`sph_phi`   [real degrees] Matlab spherical elevation angle.
            positive => rotating from horizontal (0) upwards.

EasyCap
-------
locations seem to be given in following form:
x - the angle from vertical (0) to left ear (-90)
    or to right ear (+90)
y - angle in horizontal plane - clockwise (-) or
    counter-clockwise (+)


transformation
--------------
x can be transformed to phi by:
phi = 90 - abs(x)

y (with some info from x) can be tranformed to theta
sign(x) * - 90 + y
