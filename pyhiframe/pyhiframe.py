import numpy as np
from astropy import units as u
from scipy.constants import c


class Converter(object):
    def __init__(self):
        pass


class HIConverter(object):
    def __init__(self, mode='relativistic'):
        # HI restframe
        self.nu0 = 1420.4058
        self.nu0_u = self.nu0 * u.MHz

        # full mode for Minkowski space time
        if mode.lower() == 'relativistic':
            self.v_frame = u.doppler_relativistic(self.nu0_u)
        # radio definition
        elif mode.lower() == 'radio':
            self.v_frame = u.doppler_radio(self.nu0_u)
            # velo = c * (1. - nu/nu0)

        # optical definition
        elif mode.lower() == 'optical':
            self.v_frame = u.doppler_optical(self.nu0_u)

        self.mode = mode

        return None

    # velocity-redshift conversions
    ###############################
    def velo2z(self, velo):
        """
        Converts radial velocities to redshifts

        Input
        -----
        velo : float or ndarray
            Radial velocity in km/s

        Returns
        -------
        z : float or ndarray, same shape as velo
            Redshift

        """
        if hasattr(velo, '__iter__'):
            velo = np.array(velo)

        velo *= 1.e3  # from km/s to m/s
        v_over_z = velo / c
        z = np.sqrt((1. + v_over_z) / (1. - v_over_z)) - 1.
        return z

    def z2velo(self, z):
        """
        Converts redshifts to radial velocities

        Input
        -----
        z : float or ndarray
            Redshift

        Returns
        -------
        velo : float or ndarray, same shape as z
            Radial velocity in km/s

        """
        if hasattr(z, '__iter__'):
            z = np.array(z)

        velo = (c * z**2. + 2. * c * z) / (z**2. + 2. * z + 2.)
        velo /= 1.e3  # from m/s to km/s
        return velo

    # frequency-redshift conversions
    ###############################
    def z2nu(self, z):
        """
        Converts redshifts to HI frequencies

        Input
        -----
        z : float or ndarray
            Redshift

        Returns
        -------
        nu : float or ndarray, same shape as z
            Observed frequency of HI emission in MHz

        """
        if hasattr(z, '__iter__'):
            z = np.array(z)

        nu = self.nu0 / (1. + z)

        return nu

    def nu2z(self, nu):
        """
        Converts HI frequencies to redshifts

        Input
        -----
        nu : float or ndarray
            Observed frequency of HI emission in MHz

        Returns
        -------
        z : float or ndarray, same shape as nu
            Redshift

        """
        if hasattr(nu, '__iter__'):
            nu = np.array(nu)

        z = (self.nu0 / nu) - 1.

        return z

    # velocity-frequency conversions
    ###############################
    def nu2velo(self, nu):
        """
        Converts HI frequencies to radial velocities

        Input
        -----
        nu : float or ndarray
            HI frequency

        Returns
        -------
        velo : float or ndarray, same shape as nu
            Radial velocity in km/s

        """
        if hasattr(nu, '__iter__'):
            nu = np.array(nu)

        nu_u = nu * u.MHz
        velo = nu_u.to(u.km / u.s, equivalencies=self.v_frame).value

        return velo

    def velo2nu(self, velo):
        """
        Converts radial velocities to HI frequencies

        Input
        -----
        velo : float or ndarray
            Radial velocity in km/s

        Returns
        -------
        nu : float or ndarray, same shape as nu
            HI frequency

        """
        if hasattr(velo, '__iter__'):
            velo = np.array(velo)

        velo_u = velo * u.km / u.s
        nu = velo_u.to(u.MHz, equivalencies=self.v_frame).value

        return nu
