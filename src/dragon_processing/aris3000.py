import h5py
import netCDF4
import numpy as np
import xarray as xr
from .adcp import define_rotation_nortek_XYZ_to_ENU

from .utilities import convert_matlab_datenum, decode_matlab_char, interp_datetime64


def get_beam_direction_in_nortek_instrument_coords(beam_angles,degrees=True):
    # Returns the direction aris beams in Nortek instrument coordinates assuming aris is oriented with negative beam angles pointed in x direction and positive beam angles oriented in -x direction.
    # note, this function assumes beam_angle is an array because aris beams are all at different angles.

    if degrees:
        convert = np.pi/180
    else:
        convert = 1

    return np.array([[np.sin(angle*convert),0,np.cos(angle*convert)] for angle in beam_angles])

def get_beam_direction_from_nortek_orientation_earth_coords(beam_angles,heading,roll,pitch,degrees,orientation):
    # Returns the direction of aris beams in Earth coordinates based on the beam angles and instrument orientation, assuming orientation is reported in nortek adcp format.
    # degrees is a boolean that specifies whether the angles are in degrees (True) or radians (False).
    # orientation is a string that specifies whether the nortek instrument is programmed to be oriented 'up' or 'down'.
    # note, this function assumes beam_angle is an array because aris beams are all at different angles.

    beam_directions_instrument = get_beam_direction_in_nortek_instrument_coords(beam_angles,degrees=degrees)
    rotation_to_earth = define_rotation_nortek_XYZ_to_ENU(heading,roll,pitch,degrees=degrees,orientation=orientation)
    beam_directions_earth = np.array([rotation_to_earth.apply(beamDir) for beamDir in beam_directions_instrument])
    return beam_directions_earth

def convert_aris3000(mat_file_in,frequency=3):

    attributes = {
        'beam_width_vertical' : 15,
        'beam_angle_units' : 'degrees',
        'beam_spread' : 30,

    }
    if frequency == 1.8:
        attributes['beam_width_horizontal'] = 0.3
    elif frequency == 3:
         attributes['beam_width_horizontal'] = 0.2     
    else:
        raise ValueError('frequency must be 1.8 or 3 MHz')
    
    ds = xr.Dataset()

    with h5py.File(mat_file_in,'r') as f:
        frame = f['frame']
        header = f['header']
        n_beams = int(np.squeeze(header['numbeams']))
        n_bins = int(np.squeeze(header['sampleperchannel']))

        # ds.coords['distance'] = ('bin',np.linspace(np.squeeze(header['minrange']),np.squeeze(header['maxrange']),int(np.squeeze(header['sampleperchannel']))))
        # ds.distance.attrs['units'] = 'm'
        t0 = convert_matlab_datenum(np.squeeze(header['datenum']))
        dt = ((np.arange(np.squeeze(header['numframes']))/np.squeeze(header['framerate']))*1e9).astype('timedelta64[ns]')
        ds.coords['time'] =  t0 + dt
        ds.coords['beam'] = np.arange(int(n_beams))+1
        ds.coords['bin'] = np.arange(int(n_bins))+1

        ds['min_range'] = ('time',np.full(dt.shape,np.squeeze(header['minrange'])))
        ds.min_range.attrs['units'] = 'm'
        ds['max_range'] = ('time',np.full(dt.shape,np.squeeze(header['maxrange'])))
        ds.max_range.attrs['units'] = 'm'
        ds['number_of_bins'] = ('time',np.full(dt.shape,n_bins))

        # ===== Marcel - Change =====
        # frame rate differs between recordings, so it is stored per frame rather than as a global attribute
        ds['frame_rate'] = ('time',np.full(dt.shape,np.squeeze(header['framerate'])))
        ds.frame_rate.attrs['units'] = 'Hz'
        # ===========================================

        # ===== Marcel — Memory Saving Measure 1 =====
        # It's not necessary for the data to be float64 because all values are 0 - 255. All soundings from the .mat files will be converted to uint8 here, with a check to warn if this assumption is incorrect.
        sample = frame[:, :, :10] # Take a sample from the data
        if not np.array_equal(sample, np.round(sample)) or sample.min() < 0 or sample.max() > 255: # Check if that sample has values 0 - 255 and is safe to convert to uint8
            raise ValueError(f'{mat_file_in.name}: frame values are not 0-255 integers, uint8 is unsafe')
        data = np.squeeze(frame.astype('u1')[()]) # h5py converts to uint8 while reading, so the float64 copy is never held in memory
        # ===========================================

        known_dims = {
            'time' : dt.size,
            'beam' : n_beams,
            'bin' : n_bins
        }
        dims = [next(dim for dim, n in known_dims.items() if n==s) for s in data.shape]
        ds['frame'] = (dims,data)
        # if data.shape[1] == n_beams:
        #     ds['frame'] = (('time','beam','bin'),data)
        # else:
        #     ds['frame'] = (('time','bin','beam'),data)
        ds.coords['beam_angle'] = ('beam',np.linspace(0,30,n_beams)-15)

        attributes.update({
            'version' : int(np.squeeze(header['version'])),
            'serial_number' : int(np.squeeze(header['serialnumber'])),
            'configuration' : decode_matlab_char(header['configuration'][:]),
        })

        ds.attrs = attributes

    ds = ds.transpose('time','bin','beam')

    return ds

# ===== Marcel - Memory Saving Measure 2 =====
# All files were loaded and written to the NetCDF files simultaneously. This method allows us to create the NetCDF file and write each recording individually.

def read_aris_header(mat_file):
    # Reads header of an exported .mat file
    with h5py.File(mat_file,'r') as f:
        header = f['header']
        return {
            'path' : mat_file,
            'start' : convert_matlab_datenum(np.squeeze(header['datenum'])),
            'n_frames' : int(np.squeeze(header['numframes'])),
            'n_bins' : int(np.squeeze(header['sampleperchannel'])),
            'n_beams' : int(np.squeeze(header['numbeams'])),
        }

def write_aris_depth(nc,ctd):
    # Interpolates CTD depth onto the time of each frame and writes it to an open NetCDF file as 'depth'
    # Frames outside the CTD record get NaN instead of an extrapolated depth

    times = np.asarray(nc['time'][:]).astype('datetime64[ns]') # Read frame times, stored as nanoseconds since epoch
    depth = interp_datetime64(times,ctd.time.values,ctd.depth.values,left=np.nan,right=np.nan,fill_value='constant') # Interpolate depth at each frame time
    if 'depth' not in nc.variables: # Create the variable if this file doesn't have one yet
        var = nc.createVariable('depth','f8',('time',),fill_value=np.nan) # Set per-frame depth variable
        var.units = 'meters' # Set units
        var.notes = 'CTD depth below sea surface, linearly interpolated onto the time of each frame, NaN outside the CTD record' # Add metadata
    nc['depth'][:] = depth # Write interpolated depth to frame
    n_missing = int(np.isnan(depth).sum()) # Count frames without a depth
    if n_missing: # Throw warning if any frames have no depth
        print(f'Warning: {n_missing} of {depth.size} frames are outside the CTD record and have no depth')

def write_aris_netcdf(mat_paths,nc_file,frequency,batch=256,ctd=None):
    # Writes every recording into one NetCDF file, one recording at a time, so only one recording is ever held in memory
    # The file is created at its final size from the headers, then each recording is written into its own time slice

    # Establish dimensionality of NetCDF file
    headers = sorted([read_aris_header(path) for path in mat_paths],key=lambda h: h['start']) # Read the headers of each file and sort them by time
    n_beams = {h['n_beams'] for h in headers} # Check number of beams
    if len(n_beams) > 1: # Check that all files have same beam count
        raise ValueError(f'Recordings have different numbers of beams {n_beams}. Exiting runtime.')
    n_beams = n_beams.pop() # Turn n_beams to a plain number
    n_time = sum(h['n_frames'] for h in headers) # Find total number of frames
    n_bins = max(h['n_bins'] for h in headers) # Get largest number of bins

    # Write to a temporary name and rename at the end, so a crash never leaves a complete-looking file behind
    partial_file = nc_file.with_name(nc_file.name + '.partial')
    print(f'writing {len(headers)} recordings ({n_time} frames, {n_bins} bins, {n_beams} beams) to {nc_file.name}')

    # Create the netCDF dataset
    with netCDF4.Dataset(partial_file,'w') as nc:

        # Create fields and establish parameters
        nc.createDimension('time',n_time)
        nc.createDimension('bin',n_bins)
        nc.createDimension('beam',n_beams)
        nc.createVariable('bin','i4',('bin',))[:] = np.arange(n_bins)+1
        nc.createVariable('beam','i4',('beam',))[:] = np.arange(n_beams)+1
        time = nc.createVariable('time','i8',('time',))
        time.units = 'nanoseconds since 1970-01-01'
        time.calendar = 'proleptic_gregorian'

        # Create frame variable with built-in zlib compression
        frame = nc.createVariable(
            'frame','u1',('time','bin','beam'),
            zlib=True,complevel=1,chunksizes=(16,n_bins,n_beams),fill_value=False,
        )
        frame.coordinates = 'beam_angle'
        frame.padding_value = 0
        frame.notes = 'bins beyond number_of_bins for a frame are padding designated by a 0, not data'

        # Fill the netCDF file with each dataset piecemeal
        attributes = None # Storage for global attributes
        i = 0 # Initialize indexing
        for h in headers: # For each recording...
            print(f'converting {h["path"].name}') # Print header to console
            ds = convert_aris3000(h['path'],frequency=frequency) # Proc ess data
            n = ds.time.size # Get frame count
            n_file_bins = ds.bin.size # Get bin count

            # Create per-frame variables from first file only
            if attributes is None:
                for name, var in ds.data_vars.items():
                    if var.dims == ('time',):
                        nc.createVariable(name,var.dtype.newbyteorder('='),('time',)).setncatts(var.attrs)
                nc.createVariable('beam_angle','f8',('beam',))[:] = ds.beam_angle.values
                attributes = dict(ds.attrs)

            # Check to make sure subsequent files have the first variables as the first, or else issue warning
            elif ds.attrs != attributes:
                differing = [k for k in attributes if ds.attrs.get(k) != attributes[k]]
                print(f'Warning: attributes {differing} differ from the first recording, keeping the first values')

            # Convert datetime to unix time but in nanoseconds since epoch instead of seconds
            nc['time'][i:i+n] = (ds.time.values - np.datetime64('1970-01-01')).astype('timedelta64[ns]').astype('i8')
            for name, var in ds.data_vars.items():
                if var.dims == ('time',):
                    nc[name][i:i+n] = var.values

            # Write the frames in batches with padding
            data = ds.frame.values
            for j in range(0,n,batch): # For each batch of frames...
                block = np.zeros((min(batch,n-j),n_bins,n_beams),dtype='u1') # Copy the data
                block[:,:n_file_bins,:] = data[j:j+batch]
                frame[i+j:i+j+block.shape[0]] = block # Write to file

            i += n # Advance file
            del ds, data # Free memory

        if ctd is not None: # If CTD data was given...
            print('adding CTD depth for each frame')
            write_aris_depth(nc,ctd) # Write depth for each frame

        nc.setncatts(attributes) # Write global attributes

    partial_file.replace(nc_file) # Rewrite file name to show that it's been completely and properly compiled
# ===========================================

def process_deployment_aris(deployment_path,aris_frequency,overwrite=False,ctd=None):

    aris_paths = sorted(deployment_path.rglob('*.aris'))

    # ===== Marcel - Change =====
    # Skips macOS metadata files that cause error
    aris_paths = [
        path for path in aris_paths
        if path.parent.name != 'files_that_are_not_parsing' and not path.name.startswith('._')
    ]

    # Checks that .aris data is available
    if not aris_paths:
        print('no .aris data found')
        return None
    # ===========================

    mat_paths = [path.with_suffix('.mat') for path in aris_paths]
    file_base = [path.name.split('_')[0] for path in aris_paths]
    if not np.unique(file_base).size == 1:
        raise ValueError('more than 1 aris file base name found. code cannot handle this')
    file_base = file_base[0]
    nc_file = mat_paths[0].parent / f'{file_base}_aris.nc'
    if not overwrite and nc_file.is_file():
        print(f'Skipping matlab conversion to {nc_file.name}, already converted')
        if ctd is not None: # If CTD data was given...
            with netCDF4.Dataset(nc_file,'a') as nc: # Open the existing file to append, without rewriting the frames
                if 'depth' not in nc.variables: # If depth isn't already included...
                    print(f'adding CTD depth for each frame to {nc_file.name}')
                    write_aris_depth(nc,ctd) # Write depth for each frame
    else:

        # ===== Marcel - Change =====
        # Check for missing .mat files
        missing = [path.name for path in mat_paths if not path.is_file()]
        if missing:
            print(f'{missing} do not exist. please process with matlab first')
            return None
        write_aris_netcdf(mat_paths,nc_file,aris_frequency,ctd=ctd)
        # ===========================

    # The full array can be larger than memory, data is only read when accessed
    return xr.open_dataset(nc_file)
    # ===========================
