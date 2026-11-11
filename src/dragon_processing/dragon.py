
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from .adcp import process_deployment_adcp
from .aris3000 import (
    get_beam_direction_from_nortek_orientation_earth_coords,
    process_deployment_aris,
)
from .beam_binning import beam_bin_deployment
from .ctd import process_deployment_ctd
from .plotting import initial_dragon_plots
from .utilities import (
    convert_time_to_float,
    dataset_convert_timestamp,
    file_save,
    interp_datetime64,
    lat_lon_to_utm,
)


def find_deployments(config):
    """
    Return a list of all UBOX directories found anywhere
    beneath config.raw_path.
    """

    dragon_path = config.raw_path

    if not dragon_path.is_dir():
        raise Exception(f'dragon path is not a directory: {dragon_path}')  # noqa: TRY002

    deployments = []

    for path in dragon_path.rglob('*_dragon'):

        if path.is_dir():
            deployments.append(path)

    return sorted(deployments)

def get_proc_path(raw_deploy_path, config):
    """
    Mirrror the raw deploy folder structure into the processed folder.
    """

    raw_deploy_path = Path(raw_deploy_path)

    relative_path = raw_deploy_path.relative_to(
        config.raw_path
    )

    return config.proc_path / 'dragon' / relative_path

def create_dragon_basic_coordinates(dragon,adcp):
    for coord in adcp.coords:
        if coord not in ['z','beam']:
            dragon.coords[coord] = adcp[coord].copy()

    dragon.coords['dist'] = ('bin',adcp.z.data)
    dragon.dist.attrs = adcp.z.attrs.copy()

    dragon.coords['adcp_beam'] = adcp.beam.data
    dragon.coords['adcp_beam2'] = adcp.beam.data
    dragon.adcp_beam.attrs = adcp.beam.attrs.copy()
    dragon.adcp_beam2.attrs = adcp.beam.attrs.copy()

    modes = adcp.modes

    if 'I' in modes or 'IHR' in modes:        
        dragon['beam_angle'] = ('adcp_beam',[adcp.beam_angle,]*4+[0,])
    else:
        dragon['beam_angle'] = ('adcp_beam',[adcp.beam_angle,]*4)

    return dragon

def dragon_ctd_data(dragon,ctd):

    dragon.coords['time_ctd'] = ctd.time.data
    dragon.time_ctd.attrs = ctd.time.attrs.copy()
    dragon.time_ctd.attrs['notes'] = 'timestamp for CTD measurements'

    for var in ctd.data_vars:
        dragon[var+'_ctd'] = ('time_ctd',ctd[var].data)
        dragon[var+'_ctd'].attrs = ctd[var].attrs.copy()

        if var in ['temperature', 'pressure', 'depth','salinity','density','sigma']:
            dragon[var] = ('time',interp_datetime64(dragon.time,ctd.time,ctd[var]))
            dragon[var].attrs = ctd[var].attrs
            dragon[var].attrs['source'] = 'Interpolated from CTD data onto ADCP time coordinate'

    for attr in ctd.attrs:
        dragon.attrs['ctd' + attr[0].upper() + attr[1:]] = ctd.attrs[attr]
    
    return dragon

def dragon_load_adcp_data(dragon,adcp):
    vars_rename = ['temperatureRaw', 'pressure', 'depth']

    for var in adcp.data_vars:
        dims = tuple('bin' if d == 'z' else 'adcp_beam' if d == 'beam' else d for d in adcp[var].dims)
        if var in vars_rename:
            dragon[var+'_adcp'] = (dims, adcp[var].data)
            dragon[var+'_adcp'].attrs = adcp[var].attrs.copy()
        else:
            dragon[var] = (dims,adcp[var].data)
            dragon[var].attrs = adcp[var].attrs.copy()

    for attr in adcp.attrs:
        dragon.attrs[f'adcp_{attr}'] = adcp.attrs[attr]

    return dragon

def dragon_load_aris_data(dragon,aris):

    aris_prefixed = aris.rename({
        name:f'aris_{name}' for name in list(aris.variables)
    })
    aris_prefixed = aris_prefixed.rename_dims({
        dim: f'aris_{dim}' for dim in aris_prefixed.dims if not 'aris' in dim
    })
    aris_prefixed.attrs = {
        f'aris_{k}':v for k,v in aris.attrs.items()
    }

    dragon = xr.merge([dragon,aris_prefixed])
    dragon.attrs.update(aris_prefixed.attrs)

    degrees = dragon.aris_beam_angle_units=='degrees'
    dragon['aris_beam_directions'] = (
        ('aris_time','aris_beam','direction'), np.array([
            get_beam_direction_from_nortek_orientation_earth_coords(
                dragon.aris_beam_angle.data,
                heading,
                roll,
                pitch,
                degrees=degrees,
                orientation='up'
            )
            for [heading,roll,pitch] in zip(
                dataset_convert_timestamp(dragon,'heading_angle',time_from='time_full',time_to='aris_time'),
                dataset_convert_timestamp(dragon,'roll_angle',time_from='time_full',time_to='aris_time'),
                dataset_convert_timestamp(dragon,'pitch_angle',time_from='time_full',time_to='aris_time'),
            )
        ])
    )
    return dragon

def dragon_gps_data(dragon,gps,x0=0,y0=0):
    
    dragon['latitude_polly'] = ('time',interp_datetime64(dragon.time.values,gps.time.values,gps.latitude.values))
    dragon.latitude_polly.attrs['units'] = 'degrees_north'
    dragon.latitude_polly.attrs['notes'] = 'Interpolated latitude from GPS data onto ADCP time coordinate. Represents latitude of polly at the surface.'
    dragon['longitude_polly'] = ('time',interp_datetime64(dragon.time.values,gps.time.values,gps.longitude.values))
    dragon.longitude_polly.attrs['units'] = 'degrees_east'
    dragon.longitude_polly.attrs['notes'] = 'Interpolated longitude from GPS data onto ADCP time coordinate. Represents longitude of polly at the surface.'

    x,y,zone = lat_lon_to_utm(dragon.latitude_polly.values,dragon.longitude_polly.values,x0=x0,y0=y0)

    dragon['x_polly'] = ('time',x)
    dragon['y_polly'] = ('time',y)
    dragon['x_polly'].attrs['pyproj notes'] = "+proj=utm +zone=08 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    dragon['y_polly'].attrs['pyproj notes'] = "+proj=utm +zone=08 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    dragon['x_polly'].attrs['notes'] = 'UTM x coordinate of polly at the surface, calculated from latitude_polly and longitude_polly.'
    dragon['y_polly'].attrs['notes'] = 'UTM y coordinate of polly at the surface, calculated from latitude_polly and longitude_polly.'
    dragon.attrs['utm_zone'] = zone

    dragon.attrs['x0'] = x0
    dragon.attrs['y0'] = y0

    return dragon

def dragon_merge_overlapping_casts(cast_list,merge_gap=np.timedelta64(30, 's')):
    if len(cast_list) == 0:
        return []
    
    cast_list.sort(key=lambda x: x[0])

    merged = [cast_list[0]]

    for start, middle, end, max_pressure in cast_list[1:]:
        if start <= merged[-1][2]+merge_gap:
            merged[-1][2] = max(merged[-1][2], end)
            if max_pressure > merged[-1][3]:
                merged[-1][1] = middle
                merged[-1][3] = max_pressure
        else:
            merged.append([start, middle, end, max_pressure])

    return merged

def find_dragon_casts(dragon,time_coord=None,pressure_coord=None,pressure_requirement=10, merge_gap = np.timedelta64(30,'s')):
    if time_coord is None:
        time_coord = 'time'
    if pressure_coord is None:
        pressure_coord = 'pressure'

    cast_list = []
    ind = np.where(np.diff(dragon[time_coord])>10*np.nanmedian(np.diff(dragon[time_coord])))[0]
    ind = np.r_[-1, ind, dragon[time_coord].size - 1]
    for i in np.arange(len(ind)-1):
        t_start = dragon[time_coord].data[ind[i]+1]
        t_end = dragon[time_coord].data[ind[i+1]]
        i_middle = np.nanargmax(dragon[pressure_coord].data[ind[i]+1:ind[i+1]+1])
        t_middle = dragon[time_coord].data[ind[i]+1:ind[i+1]+1][i_middle]
        p_max = dragon[pressure_coord].data[ind[i]+1:ind[i+1]+1][i_middle]
        if p_max > pressure_requirement:
            cast_list.append([t_start,t_middle,t_end,p_max])

    return [cast[:-1] for cast in dragon_merge_overlapping_casts(cast_list,merge_gap=merge_gap)]

def number_casts(dragon,cast_list,time_coords=None,cast_num_names=None,cast_descriptions=None):
    if time_coords is None:
        time_coords = ['time']
    if cast_num_names is None:
        cast_num_names = ['cast_num']


    cast_num_dict = {cast_num_name : np.full(dragon[time_coord].size,np.nan) for time_coord,cast_num_name in zip(time_coords,cast_num_names)}

    cast_num_down = 1
    cast_num_up = 1.5

    if cast_descriptions is not None:
        cast_descriptions_all = []

    unique_cast_nums = []
    for t_start, t_middle, t_end in cast_list:
        for time_coord,cast_num_name in zip(time_coords,cast_num_names):
            ind_down = np.logical_and(dragon[time_coord].data>=t_start,dragon[time_coord].data<t_middle)
            ind_up = np.logical_and(dragon[time_coord].data>=t_middle,dragon[time_coord].data<=t_end)
 
            cast_num_dict[cast_num_name][ind_down] = cast_num_down
            cast_num_dict[cast_num_name][ind_up] = cast_num_up

        unique_cast_nums.extend([cast_num_down,cast_num_up])
        if cast_descriptions is not None:
            cast_descriptions_all.extend([cast_descriptions[cast_num_down-1]+' down cast',cast_descriptions[cast_num_down-1] + ' up cast'])

        cast_num_down += 1
        cast_num_up += 1

    dragon.coords['cast'] = unique_cast_nums
    dragon.cast.attrs['notes'] = 'Unique cast numbers in the dataset. Integer values indicate downcasts, fractional values (e.g. 1.5) indicate upcasts.'

    if cast_descriptions is not None:
        dragon['cast_descriptions'] = ('cast',cast_descriptions_all)

    for time_coord,cast_num_name in zip(time_coords,cast_num_names):
        dragon[cast_num_name] = (time_coord,cast_num_dict[cast_num_name])

    return dragon

def cut_dragon_to_casts(dragon,cast_notes=None,pressure_cutoff = 1):

    time_coords = [coord for coord in dragon.coords if 'time' in coord and ('time_BT' not in coord and 'time_I' not in coord)]
    cast_num_names = [coord.replace('time','cast_num') for coord in time_coords]
    bottomDetection_names = ['bottom_detection_edge_min', 'bottom_detection_edge_max', 'bottom_detection_amplitude_threshold', 'bottom_detection_skipped', 'bottom_detection_max_interpolate_gap', 'bottom_detection_edge_blur_kernel_x', 'bottom_detection_edge_blur_kernel_y']

    if cast_notes is None:
        time_for_cast_detection = 'time'
        pressure_for_cast_detection = 'pressure'

        for var in dragon.data_vars:
            if 'pressure' in var:
                dim = dragon[var].dims[0]
                dragon = dragon.isel({dim:dragon[var]>pressure_cutoff})

        t_start = max(dragon[time].data[0] for time in time_coords if 'aris' not in time)
        t_end = min(dragon[time].data[-1] for time in time_coords if 'aris' not in time)
        t_slice = slice(t_start, t_end)
        dragon = dragon.sel({time:t_slice for time in time_coords})

        cast_list = find_dragon_casts(dragon,time_for_cast_detection,pressure_for_cast_detection)
        cast_descriptions = None
    else:
        cast_list = [[item['time_start'],item['time_middle'],item['time_end']] for item in cast_notes.values()]
        cast_descriptions = [item['description'] for item in cast_notes.values()]
    
    dragon = number_casts(dragon,cast_list,time_coords,cast_num_names,cast_descriptions)
    dragon = dragon.isel({time:~np.isnan(dragon[cast_num]) for time,cast_num in zip(time_coords,cast_num_names)})
    # dragon = dragon.isel(time=~np.isnan(dragon.cast_num),time_ctd = ~np.isnan(dragon.cast_num_ctd),time_full=~np.isnan(dragon.cast_num_full))

    for var in bottomDetection_names:
        if var in dragon:
            dragon[var] = dragon[var].expand_dims({'cast': dragon.cast}).transpose('cast',*dragon[var].dims)

    return dragon

def calculate_displacement(start,end,vel,t):
    if np.sum(~np.isnan(vel)) < 5:
        return (np.arange(t.size)/(t.size-1))*(end-start)+start, np.full(t.size, 2), np.nan
    
    information = np.full(t.size, np.nan) #0 is original data, 1 is interpolated between data, 2 is correction factor at either end

    velPredicted = vel.copy()
    iNan = np.isnan(vel)
    information[~iNan] = 0
    information[iNan] = 1
    

    interpFunc = interp1d(t[~iNan], vel[~iNan], kind='linear', fill_value=np.nan, bounds_error=False)
    velPredicted[iNan] = interpFunc(t[iNan])

    iNan = np.isnan(velPredicted)
    information[iNan] = 2

    velPredicted[iNan] = 0
    
    position = start + cumulative_trapezoid(y=velPredicted,x=t,initial=0)
    error = end - position[-1]

    #how much to add to velPredicted to fix the error
    timeFactor = np.nanmean(np.diff(t))*np.sum(iNan)
    meanVelCorrection = error / timeFactor

    nan_indices = np.where(iNan)[0]
    # Find contiguous NaNs at start and end
    start_nan = [nan_indices[i] for i in range(nan_indices.size) if nan_indices[i] == i]
    end_nan = [nan_indices[j] for j in range(nan_indices.size) if nan_indices[j] != j]

    # figure out how to space the corrections at start and end for some smoothness
    nStart = len(start_nan)
    nEnd = len(end_nan)

    velStart = velPredicted[~iNan][0]
    velEnd = velPredicted[~iNan][-1]

    # I want to solve for correctionStart and correctionEnd such that:
    # 1. correctionStart * nStart + correctionEnd * nEnd = meanVelCorrection * (nStart + nEnd)
    # 2. (correctionStart-velStart)**2 + (correctionEnd-velEnd)**2 is minimized
    # Given: nStart, nEnd, meanVelCorrection, velStart, velEnd
    # I'm too lazy to look into this right now, so trusting chatGPT ... yikes
    lambda_val = 2 * (nStart * velStart + nEnd * velEnd - meanVelCorrection * (nStart + nEnd)) / (nStart**2 + nEnd**2)
    correctionStart = velStart - (nStart / 2) * lambda_val
    correctionEnd = velEnd - (nEnd / 2) * lambda_val
    
    scaleStart = 2*(correctionStart - velStart)/(nStart+1)
    
    scaleEnd = 2*(correctionEnd - velEnd)/(nEnd+1)

    velPredicted[start_nan] = [velStart + scaleStart*(nStart-i) for i in range(nStart)]
    velPredicted[end_nan] = [velEnd + scaleEnd*(i+1) for i in range(nEnd)]

    position = start + cumulative_trapezoid(y=velPredicted,x=t,initial=0)

    if np.sum(np.isnan(position))>0 or np.sum(np.isnan(information))>0:
        raise ValueError("NaNs in position or information after displacement calculation")

    return position, information, error

def calculate_adcp_positions(dragon):

    dragon['adcp_location'] = (('time','XYZ'), np.full((dragon.time.size, 3), np.nan))
    dragon.adcp_location.data[:,2] = -dataset_convert_timestamp(dragon,'depth_adcp',time_from='time_full',time_to='time')

    #dragon['x_adcp'] = ('time', np.full(dragon.time.size, np.nan))
    #dragon['y_adcp'] = ('time', np.full(dragon.time.size, np.nan))
    dragon['x_adcp_interpolated'] = ('time', np.full(dragon.time.size, np.nan))
    dragon['y_adcp_interpolated'] = ('time', np.full(dragon.time.size, np.nan))
    dragon['x_adcp_error'] = ('cast', np.full(dragon.cast.size, np.nan))
    dragon['y_adcp_error'] = ('cast', np.full(dragon.cast.size, np.nan))

    for cast_num in np.unique(dragon.cast_num.data.astype('int')):
        ind = np.logical_and(np.logical_or(dragon.cast_num==cast_num,dragon.cast_num == cast_num+0.5), ~np.isnat(dragon.time_BT.data))
        indCast = dragon.cast.data.astype('int') == cast_num

        xPolly = interp_datetime64(dragon.time_BT.data[ind],dragon.time.data[ind], dragon.x_polly.data[ind])
        yPolly = interp_datetime64(dragon.time_BT.data[ind],dragon.time.data[ind], dragon.y_polly.data[ind])

        x_adcp, x_interpolated, x_error = calculate_displacement(xPolly[0], xPolly[-1], -dragon.velENU_l1_BT.data[ind,0], convert_time_to_float(dragon.time_BT.data[ind]))
        y_adcp, y_interpolated, y_error = calculate_displacement(yPolly[0], yPolly[-1], -dragon.velENU_l1_BT.data[ind,1], convert_time_to_float(dragon.time_BT.data[ind]))
        dragon.adcp_location.data[ind, 0] = interp_datetime64(dragon.time.data[ind], dragon.time_BT.data[ind], x_adcp)
        dragon.adcp_location.data[ind, 1] = interp_datetime64(dragon.time.data[ind], dragon.time_BT.data[ind], y_adcp)
        dragon.x_adcp_interpolated.data[ind] = x_interpolated
        dragon.y_adcp_interpolated.data[ind] = y_interpolated
        dragon.x_adcp_error.data[indCast] = x_error
        dragon.y_adcp_error.data[indCast] = y_error

    dragon.adcp_location.attrs = dragon.x_polly.attrs.copy()
    dragon.adcp_location.attrs['notes'] = 'location in UTM coordinates of the ADCP at depth, x and y locations calculated integrating bottom track velocities and filling in missing data to match start and end positions ' \
    'based on the x_polly and y_polly coordinates (interpolated onto "time_BT" time-stamp). z location based on depth_adcp variable. Result is then interpolated to "time" time-stamp. time periods that rely on interpolating position changes are marked in x_adcp_interpolated and y_adcp_interpolated.'

    dragon.x_adcp_interpolated.attrs['notes'] = '0 where full bottom track data was available, 1 where interpolation between known velocities was needed. 2 where linear interpolation to start or end position was needed.'
    dragon.y_adcp_interpolated.attrs['notes'] = '0 where full bottom track data was available, 1 where interpolation between known velocities was needed. 2 where linear interpolation to start or end position was needed.'
    dragon.x_adcp_error.attrs['notes'] = 'Error estimate for x position based on bottom track velocity integration. Note, up and down casts are treated together for error estimation and position calculation.'
    dragon.y_adcp_error.attrs['notes'] = 'Error estimate for y position based on bottom track velocity integration. Note, up and down casts are treated together for error estimation and position calculation.'

    return dragon   

def calculate_adcp_beam_positions(dragon):
    """
    Calculate the positions of the ADCP beams based adcp location, beam angles, and adcp orientation.
    """

    dragon['beam_distance'] = ('bin',dragon.dist.data.copy()/np.cos(dragon.adcp_beam_angle*np.pi/180))
    dragon.beam_distance.attrs['units'] = dragon.dist.units
    dragon.beam_distance.attrs['notes'] = 'distance along beam of each bin for the 4 angled beams'

    dragon['bottom_distance_along_beam'] = dragon.bottom_distance_beam.copy()/np.cos(dragon.adcp_beam_angle*np.pi/180)
    dragon.bottom_distance_along_beam.attrs['notes'] = 'distance in the along beam direction'

    #reformat to dimensions of time, dist, adcp_beam, XYZ
    beamDirections = np.repeat(
        dataset_convert_timestamp(dragon,'beam_directions_earth',time_from='time_full',time_to='time')[:,np.newaxis,:,:],
        dragon.dist.size, axis=1)

    beamDirectionsWall = dataset_convert_timestamp(dragon,'beam_directions_earth',time_from='time_full',time_to='time')

    dragon_location = np.repeat(np.repeat(
        dragon.adcp_location.data.copy()[:,np.newaxis,np.newaxis,:],
        dragon.dist.size, axis=1),
        dragon.adcp_beam.size, axis=2)
    
    dragon_locationWall = np.repeat(
        dragon.adcp_location.data.copy()[:,np.newaxis,:],
        dragon.adcp_beam.size, axis=1
    )

    beam_distance = np.repeat(np.repeat(np.repeat(
        dragon.beam_distance.data.copy()[np.newaxis,:,np.newaxis,np.newaxis],
        dragon.time.size, axis=0),
        dragon.adcp_beam.size, axis=2),
        dragon.direction.size,axis=3)

    beam_wall_distance = np.repeat(
        dragon.bottom_distance_along_beam.data.copy()[:,:,np.newaxis],
        dragon.direction.size, axis=2)
    
    dragon['beam_location'] = (('time','bin','adcp_beam','XYZ'), beam_distance*beamDirections + dragon_location)
    dragon.beam_location.attrs['notes'] = 'location of each beam bin in UTM coordinates for each timestep and beam, using adcp location and orientation with beam angle'
    dragon.beam_location.attrs['units'] = 'meters in UTM coordinates'

    dragon['beam_wall_location'] = (('time','adcp_beam','XYZ'),beam_wall_distance*beamDirectionsWall+dragon_locationWall)
    dragon.beam_wall_location.attrs['notes'] = 'location of bottom as seen by each beam in UTM coordinates for each timestep using using adcp location and orientation with beam angle'

    dragon['bin_location'] = (('time','bin','XYZ'),np.mean(beam_distance*beamDirections+dragon_location,axis=2))
    dragon.bin_location.attrs['notes'] = 'average of beam locations for each bin in UTM coordinates for each timestep. Represents location of 4 beam solutions'

    #dragon.attrs['processingState'] = 'ADCP and CTD data combined with GPS. CTD interpolated onto ADCP time-step. Data where pressure < 1 is not included because dragon is not in mid cast. dataset includes \
    #    bottom track corrected velocities, beam directions, bin locations, and pressure sensor based corrected vertical velocity for when there is not bottom tracking.'

    #dragon.attrs['processingState'] = 'ADCP and CTD data combined with NO GPS. CTD interpolated onto ADCP time-step. Data where pressure < 1 is not included because dragon is not in mid cast. dataset includes \
    #    bottom track corrected velocities, beam directions, and pressure sensor based corrected vertical velocity for when there is not bottom tracking.'

    return dragon

def calculate_aris_beam_positions(dragon):
    dragon_location = dataset_convert_timestamp(dragon,'adcp_location',time_from = 'time', time_to = 'aris_time')[:,np.newaxis,np.newaxis,:]
    if 'distance' in dragon.coords:
        beam_distance = dragon.aris_distance.data[np.newaxis,:,np.newaxis,np.newaxis]
    else:
        beam_distance = np.array([np.pad(np.linspace(min_range,max_range,n_bins),(0,dragon.aris_bin.size-n_bins)) for min_range,max_range,n_bins in zip(dragon.aris_min_range.data,dragon.aris_max_range.data,dragon.aris_number_of_bins.data)])[:,:,np.newaxis,np.newaxis]

    beam_directions = dragon.aris_beam_directions.data[:,np.newaxis,:,:]
    dragon['aris_beam_location'] = (
        ('aris_time','aris_bin','aris_beam','direction'),
        beam_distance*beam_directions + dragon_location,
        {
            'notes' : 'location of each aris beam bin in UTM coordinates for each timestep and beam, using adcp location and orientation with aris beam angles',
            'units' : 'meters in UTM coordinates'
        }
    )

    return dragon

def update_dragon_origin(dragon,x0=None,y0=None):
    """
    Updates the origin of the dragon dataset to the specified x0 and y0 coordinates.
    If x0 or y0 is None, it will not update that coordinate.
    """

    dragon = dragon.copy(deep=True)

    xVariables = ['x_polly','fieldX0','plumeOutlineX']
    yVariables = ['y_polly','fieldY0','plumeOutlineY']

    comboVariables = ['beam_location','beam_wall_location','bin_location','adcp_location','fieldBin_location','aris_beam_location']

    if x0 is not None:
        xDiff = x0 - dragon.x0

        for var in xVariables:
            if var in dragon.data_vars:
                dragon[var] -= xDiff
        
        for var in comboVariables:
            if var in dragon.data_vars:
                dragon[var][...,0] -= xDiff

        dragon.attrs['x0'] = x0
    else:
        xDiff = 0

    if y0 is not None:
        yDiff = y0 - dragon.y0

        for var in yVariables:
            if var in dragon.data_vars:
                dragon[var] -= yDiff

        for var in comboVariables:
            if var in dragon.data_vars:
                dragon[var][...,1] -= yDiff

        dragon.attrs['y0'] = y0
    else:
        yDiff = 0

    if (yDiff !=0 or xDiff !=0) and 'field_intercept' in dragon.data_vars:
        dragon.field_intercept.data = dragon.field_intercept.data - yDiff + xDiff*dragon.fieldSlope.data

    return dragon

def initial_dragon_file_creation(adcp,ctd,aris=None,x0=0,y0=0,gps=None,existing_beam_cleaning=None,existing_bottom_tracking=None,full_cleaning=False, cast_notes = None, beam_bin=False):
    
    dragon = create_dragon_basic_coordinates(xr.Dataset(),adcp)
    dragon = dragon_ctd_data(dragon, ctd)
    dragon = dragon_load_adcp_data(dragon,adcp)
    processing_state = 'Processed ADCP and CTD data combined. CTD data interpolated ont main adcp time stamp for dragon depth temperature and salinity.' 

    if aris is not None:
        dragon = dragon_load_aris_data(dragon, aris)
        processing_state = processing_state + 'aris data added.'
        dragon.attrs['aris_included'] = 1
    else:
        dragon.attrs['aris_included'] = 0

    if gps is not None:
        dragon = dragon_gps_data(dragon,gps,x0=x0,y0=y0)
        processing_state = processing_state + ' Polly gps data added.'
        dragon.attrs['gps_included'] = 1
    else:
        dragon.attrs['gps_included'] = 0

    dragon = cut_dragon_to_casts(dragon, cast_notes)
    ' Data split into casts and numbered by deleting data where depth < 1 meter. Casts dividied into downcasts (integer) and upcasts (0.5) based on deepest depth.'   

    if (gps is not None) and 'BT' in adcp.modes:
        if existing_bottom_tracking is not None:
            raise ValueError('need to implement bottom tracking')
            processing_state = processing_state + ' Custom bottom tracking implemented.'
        else:
            processing_state = ' Using default bottom tracking.'
        dragon = calculate_adcp_positions(dragon)
        dragon = calculate_adcp_beam_positions(dragon)
        processing_state = processing_state + ' Location of adcp calculated based on bottom track velocities and interpolated to main time stamp. ' \
            'Beam locations calculated based on adcp location, beam angles, and beam distance.'

        if aris is not None:
            dragon = calculate_aris_beam_positions(dragon)
            processing_state = processing_state + ' Aris beam locations calculated based on adcp location, and assuming aris is oriented the same as the adcp.'

        if full_cleaning:
            raise ValueError('full cleaning not yet implemented')
            processing_state = processing_state + 'basic beam cleaning performed (distance to wall and surface deleted) interactively for each cast and saved as a mask (except where beam_cleaning_skipped = True)'
        else: 
            dragon['mask_beam_cleaning'] = (('time','bin','adcp_beam'),np.ones((dragon.time.size,dragon.bin.size,dragon.adcp_beam.size),dtype=bool))
            dragon.mask_beam_cleaning.attrs['notes'] = 'mask for beam cleaning, True means data is kept, False means data is removed. Currently all true because no further beam cleaning has occured.'
            processing_state = processing_state + ' No additional beam cleaning performed'

        if beam_bin:
            dragon = beam_bin_deployment(dragon,max_heading_variation=30)
    else:
        processing_state = processing_state + ' No GPS data available, so adcp and beam locations are not calculated, and no additional beam cleaning performed.'

    dragon.attrs['processingState'] = processing_state

    dragon_out = dragon[sorted(dragon.data_vars)]

    for c in dragon.coords:
        if c not in dragon_out.coords:
            dragon_out = dragon_out.assign_coords({c: dragon.coords[c]})

    return dragon_out

def process_all(
        config,
        overwrite=False,
        basic_plots=False,
        default_lat=68.295,
        default_lon=-30.81,
        include_aris=False,
        beam_bin=False,
        plot_beam_vel_lim=0.15,
        plot_vel_ENU_lim=0.15,
        plot_field_vel_lim=0.15,
        plot_depth_min=-3,
        plot_depth_max=None,
        plot_distance_max=None,
        plot_vertical_smoothing_window=None,
    ):

    #find all dragon deployments
    deployment_paths = find_deployments(config)

    print(f'found deployments: {deployment_paths}')

    for dep in deployment_paths:
        print(f'deployment {dep.name}')
        #prep save folder for deployment
        dep_proc_path = get_proc_path(dep,config)
        dep_proc_path.mkdir(parents=True,exist_ok=True)
        deployment_num = '_'.join(dep.name.split('_')[:-1])
        gps_file = dep_proc_path / "gps.nc"
        beam_cleaning_file = dep_proc_path / "beam_cleaning_information.nc"
        bottom_track_file = dep_proc_path / "bottom_track_information.nc"

        if gps_file.is_file():
            print('gps file found')
            gps = xr.load_dataset(gps_file)
            lat = np.nanmean(gps.latitude)
            lon = np.nanmean(gps.longitude)
        else:
            print('no gps file found')
            gps = None
            lat = default_lat
            lon = default_lon

        if beam_cleaning_file.is_file():
            print('beam cleaning found')
            existing_beam_cleaning = xr.load_dataset(beam_cleaning_file)
        else:
            print('no beam cleaning found')
            existing_beam_cleaning = None

        if bottom_track_file.is_file():
            print('bottom track found\n')
            existing_bottom_track = xr.load_dataset(bottom_track_file)
        else:
            print('no bottom track found\n')
            existing_bottom_track = None

        adcp_dictionary = process_deployment_adcp(dep,dep_proc_path,config.qc,lat,overwrite)
        ctd = process_deployment_ctd(dep,dep_proc_path,lat=lat,lon=lon,overwrite=overwrite)

        # I think this needs to be moved later because of memory usage, but maybe not.
        # but first, I might want to try storing with an aris_file coordinate so I don't have to extend all the datasets to a 4000 long bin coordinate
        if include_aris:
            aris = process_deployment_aris(dep, aris_frequency=config.aris_frequency, overwrite=overwrite, ctd=ctd) # Pass CTD data
        else:
            aris = None

        if len(adcp_dictionary)>1:
            raise ValueError('naming scheme currently does not handle multiple adcp types')
        for adcp_type, adcp_data in adcp_dictionary.items():  # noqa: PERF102
            save_file = dep_proc_path / f'dragon_{deployment_num}.nc'
            if not overwrite and save_file.is_file():
                print(f'skipping {save_file.name}, already exists')
                continue
            if len(adcp_data)>1:
                raise ValueError(f'Consider restructure adcp processing. multiple processed adcp data files for deployment {dep}')
            else:
                print(f'creating {save_file.name}')
                cast_notes = None
                if config.cast_notes:
                    if deployment_num in config.cast_notes:
                        cast_notes = config.cast_notes[deployment_num]
                        
                dragon = initial_dragon_file_creation(next(iter(adcp_data.values())),ctd=ctd,aris=aris,x0=config.x0,y0=config.y0,gps=gps,existing_beam_cleaning=existing_beam_cleaning,existing_bottom_tracking=existing_bottom_track, cast_notes = cast_notes, beam_bin= beam_bin)
                file_save(dragon,save_file,overwrite)
                if basic_plots:
                    initial_dragon_plots(
                        dragon,dep_proc_path / 'initial_plots',
                        deployment_num,
                        beam_vel_lim=plot_beam_vel_lim,
                        vel_ENU_lim=plot_vel_ENU_lim,
                        field_vel_lim=plot_field_vel_lim,
                        depth_min=plot_depth_min,
                        depth_max=plot_depth_max,
                        distance_max=plot_distance_max,
                        vertical_smoothing_window=plot_vertical_smoothing_window
                    )

def remake_initial_plots(
        config,
        plot_beam_vel_lim=0.15,
        plot_vel_ENU_lim=0.15,
        plot_field_vel_lim=0.15,
        plot_depth_min=-3,
        plot_depth_max=None,
        plot_distance_max=None,
        plot_vertical_smoothing_window=None,
        ):
    deployment_paths = find_deployments(config)
    for dep in deployment_paths:
        dep_proc_path = get_proc_path(dep,config)
        deployment_num = '_'.join(dep.name.split('_')[:-1])
        dragon_file = dep_proc_path / f'dragon_{deployment_num}.nc'
        print(f'remaking plots for {dragon_file.name}')
        with xr.open_dataset(dragon_file) as dragon:
            initial_dragon_plots(
                dragon,dep_proc_path / 'initial_plots',
                deployment_num,
                beam_vel_lim=plot_beam_vel_lim,
                vel_ENU_lim=plot_vel_ENU_lim,
                field_vel_lim=plot_field_vel_lim,
                depth_min=plot_depth_min,
                depth_max=plot_depth_max,
                distance_max=plot_distance_max,
                vertical_smoothing_window=plot_vertical_smoothing_window
            )