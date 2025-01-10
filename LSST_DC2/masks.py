import numpy as np
import healpy as hp
import fitsio
from ssf import load_ssf
from collections import OrderedDict as odict

def ang2pix(nside,lon,lat,nest=False):
    """
    Input (lon, lat) in degrees instead of (theta, phi) in radians
    """
    theta = np.radians(90. - lat)
    phi = np.radians(lon)
    return hp.ang2pix(nside, theta, phi, nest=nest)
    

def read_map(filename,nest=False,hdu=None,h=False,verbose=True):
    """Read a healpix map from a fits file.  Partial-sky files,
    if properly identified, are expanded to full size and filled with UNSEEN.
    Uses fitsio to mirror much (but not all) of the functionality of healpy.read_map

    Parameters:
    -----------
    filename : str
      the fits file name
    nest : bool, optional
      If True return the map in NEST ordering, otherwise in RING ordering;
      use fits keyword ORDERING to decide whether conversion is needed or not
      If None, no conversion is performed.
    hdu : int, optional
      the header number to look at (start at 0)
    h : bool, optional
      If True, return also the header. Default: False.
    verbose : bool, optional
      If True, print a number of diagnostic messages

    Returns:
     -------
    m [, header] : array, optionally with header appended
      The map read from the file, and the header if *h* is True.
    """

    data,hdr = fitsio.read(filename,header=True,ext=hdu)

    nside = int(hdr.get('NSIDE'))
    if verbose: print('NSIDE = {0:d}'.format(nside))

    if not hp.isnsideok(nside):
        raise ValueError('Wrong nside parameter.')
    sz=hp.nside2npix(nside)

    ordering = hdr.get('ORDERING','UNDEF').strip()
    if verbose: print('ORDERING = {0:s} in fits file'.format(ordering))

    schm = hdr.get('INDXSCHM', 'UNDEF').strip()
    if verbose: print('INDXSCHM = {0:s}'.format(schm))
    if schm == 'EXPLICIT':
        partial = True
    elif schm == 'IMPLICIT':
        partial = False
        # monkey patch on a field method
    fields = data.dtype.names

    # Could be done more efficiently (but complicated) by reordering first
    if hdr['INDXSCHM'] == 'EXPLICIT':
        m = hp.UNSEEN*np.ones(sz,dtype=data[fields[1]].dtype)
        m[data[fields[0]]] = data[fields[1]]
    else:
        m = data[fields[0]].ravel()

    if (not hp.isnpixok(m.size) or (sz>0 and sz != m.size)) and verbose:
        print('nside={0:d}, sz={1:d}, m.size={2:d}'.format(nside,sz,m.size))
        raise ValueError('Wrong nside parameter.')
    if not nest is None:
        if nest and ordering.startswith('RING'):
            idx = hp.nest2ring(nside,np.arange(m.size,dtype=np.int32))
            if verbose: print('Ordering converted to NEST')
            m = m[idx]
            return  m[idx]
        elif (not nest) and ordering.startswith('NESTED'):
            idx = hp.ring2nest(nside,np.arange(m.size,dtype=np.int32))
            m = m[idx]
            if verbose: print('Ordering converted to RING')

    if h:
        return m, header
    else:
        return m


def load_survey_masks(surveys=['des','ps1']):
    masks = {}
    ssfs = {}
    for survey in surveys:
        masks[survey] = load_mask(survey)
        ssfs[survey] = load_ssf(survey)
    return masks, ssfs


def load_mask(survey):
    """
    Loads selection masks from https://arxiv.org/abs/1912.03302 to simulation data

    Args:
        survey (string): name of survey

    Returns:
        indices (bitmask): bitmask for a given survey
    """
    if survey == 'des':
        indices = read_map('../Classifier/healpix_mask_{}_v5.1.fits'.format(survey), nest=True)
    elif survey == 'ps1':
        indices = read_map('../Classifier/healpix_mask_{}_v5.1.fits'.format(survey), nest=True)
    elif survey.startswith('lsst'):
        indices = read_map('../Classifier/healpix_mask_{}_v2.fits'.format('lsst'), nest=True)
    else:
        print('invalid survey')
    return indices


def evaluate_mask(ra,dec,indices,survey,cut_footprint=True,cut_ebv=True,cut_associate=True,cut_bsc=True,cut_dwarfs=True,dwarftype='2'):
    """
    Applies selection masks from https://arxiv.org/abs/1912.03302 to simulation data

    Args:
        ra (array): array of ra values (in degrees) for mock satellites 
        dec (array): array of decd values (in degrees) for mock satellites 
        indices (bitmask): bitmask for a given survey
        survey (string): name of survey

    Returns:
        survey_flags (Boolean array): array of Boolean values (True = good pixel, False = bad pixel) 
    """
    pix = ang2pix(4096, ra, dec, nest=True)

    BITS = odict([
    ('GOOD',   0b000000000), # No flags
    ('DWARF4', 0b000000001), # near known dwarf (type2 >= 4)
    ('DWARF3', 0b000000010), # near known dwarf (type2 == 3)
    ('DWARF2', 0b000000100), # near known dwarf (type2 == 2)
    ('ASSOC',  0b000001000), # near object in catalogs (excluding DWARFs)
    ('STAR',   0b000010000), # near bright star
    ('EBV',    0b000100000), # E(B-V) > 0.2
    ('FOOT',   0b001000000), # outside of footprint
    ('FAIL',   0b010000000), # location of failure in ugali PS1 processing
    ('ART',    0b100000000), # location of artifact in PS1 footprint
])

    if cut_footprint:
        cut_footprint_flags = np.where(indices[pix] & BITS['FOOT'], False, True)
        survey_flags = cut_footprint_flags

    if cut_ebv:
        cut_ebv_flags = np.where(indices[pix] & BITS['EBV'], False, True)
        survey_flags = survey_flags & cut_ebv_flags

    if cut_associate:
        cut_associate_flags = np.where(indices[pix] & BITS['ASSOC'], False, True)
        survey_flags = survey_flags & cut_associate_flags

    if cut_bsc:
        cut_bsc_flags = np.where(indices[pix] & BITS['STAR'], False, True)
        survey_flags = survey_flags & cut_bsc_flags

    if cut_dwarfs:
        cut_dwarf_flags = np.where(indices[pix] & BITS['DWARF{}'.format(dwarftype)], False, True)
        survey_flags = survey_flags & cut_dwarf_flags

    if survey == 'ps1':
        cut_fail_flags = np.where(indices[pix] & BITS['FAIL'], False, True)
        cut_art_flags = np.where(indices[pix] & BITS['ART'], False, True)
        survey_flags = survey_flags & cut_fail_flags & cut_art_flags

    return survey_flags