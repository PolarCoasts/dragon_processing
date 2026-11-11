import gsw
import numpy as np
import pyrsktools
import xarray as xr

from .utilities import file_save


def readRSK(rskFile):
    # Reads a RSK file and returns a xarray Dataset with the CTD data.

    ctd = xr.Dataset()

    with pyrsktools.RSK(rskFile) as rsk:
        rsk.readdata()

        ctd.coords['time'] = rsk.data['timestamp'].astype('datetime64[ns]')
        ctd.time.attrs['timezone'] = 'UTC'

        ctd['temperature'] = ('time',rsk.data['temperature'])
        ctd.temperature.attrs['units'] = 'Celsius'

        ctd['conductivity'] = ('time',rsk.data['conductivity'])
        ctd.conductivity.attrs['units'] = 'mS/cm'

        ctd['pressure'] = ('time',rsk.data['pressure'])
        ctd.pressure.attrs['units'] = 'dbar'

        ctd['sea_pressure'] = ('time',rsk.data['sea_pressure'])
        ctd.sea_pressure.attrs['units'] = 'dbar'
        ctd.sea_pressure.attrs['notes'] = 'calculated by pyrsk tools using TEOS-10 GSW conversion and assuming a default atmospheric pressure of 10.1325 dbar'

        ctd['depth'] = ('time',rsk.data['depth'])
        ctd.depth.attrs['units'] = 'meters'
        ctd.depth.attrs['notes'] = 'calculated by pyrsk tools using TEOS-10 GSW conversion relying on the default sea_pressure calculation'

        ctd['salinity'] = ('time',rsk.data['salinity'])
        ctd.salinity.attrs['units'] = 'PSU'
        ctd.salinity.attrs['notes'] = 'calculated by pyrsk tools using TEOS-10 GSW conversion relying on the default sea_pressure calculation'

        ctd.attrs['deploymentName'] = rsk.deployment.name
        ctd.attrs['instrumentID'] = rsk.instrument.serialID
        ctd.attrs['instrumentModel'] = rsk.instrument.model
        ctd.attrs['instrumentPartNumber'] = rsk.instrument.partNumber
    
    return ctd

def initial_clean_dragon_ctd(ctd, lat, lon):
    """
    cleaning and processing the CTD data from the Dragon before incorporating it into the Dragon dataset.

    This function assumes that the CTD data is in a xarray Dataset format with the following variables:
        temperature: in-situ temperature in Celsius
        conductivity: raw conductivity in mS/cm
        pressure: raw pressure in dbar

    First, this function corrects the pressure for air pressure effects.

    Then, it calculates the following new variables using the TEOS-10 GSW library:
        depth: depth below sea surface in meters
        salinity: absolute salinity in g/kg
        temperature: conservative temperature in Celsius
        density: in-situ density anomaly in kg/m^3
        sigma: potential density anomaly (sigma0) in kg/m^3

    """

    ctdNew = xr.Dataset(coords=ctd.coords, attrs=ctd.attrs)

    ctdNew['temperature_raw'] = ctd.temperature.copy()
    ctdNew.temperature_raw.attrs['notes'] = 'In-situ temperature'

    ctdNew['conductivity'] = ctd.conductivity.copy()
    ctdNew.conductivity.attrs['notes'] = 'raw conductivity'

    pressure = ctd.pressure.copy()
    ctdNew['pressure'] = pressure-np.mean(pressure[pressure<np.min(pressure)+1])
    ctdNew.pressure.attrs['units'] = 'dbar'
    ctdNew.pressure.attrs['notes'] = 'sea pressure corrected for air pressure'
    
    ctdNew['depth'] = np.abs(gsw.conversions.z_from_p(ctdNew.pressure,lat))
    ctdNew.depth.attrs['units'] = 'meters'
    ctdNew.depth.attrs['notes'] = 'depth below sea surface'

    ctdNew['salinity'] = gsw.conversions.SA_from_SP(
        gsw.conversions.SP_from_C(
            ctdNew.conductivity, 
            ctdNew.temperature_raw, 
            ctdNew.pressure
        ),
        ctdNew.pressure,
        lon,
        lat
    )
    ctdNew.salinity.attrs['units'] = 'g/kg'
    ctdNew.salinity.attrs['notes'] = 'absolute salinity'

    ctdNew['temperature'] = gsw.conversions.CT_from_t(
        ctdNew.salinity,
        ctdNew.temperature_raw,
        ctdNew.pressure
    )
    ctdNew.temperature.attrs['units'] = 'Celsius'
    ctdNew.temperature.attrs['notes'] = 'conservative temperature'

    ctdNew['density'] = gsw.density.rho(
        ctdNew.salinity,ctdNew.temperature,ctdNew.pressure
    )
    ctdNew.density.attrs['units'] = 'kg/m^3'
    ctdNew.density.attrs['notes'] = 'In-situ density'

    ctdNew['sigma'] = gsw.density.sigma0(
        ctdNew.salinity,ctdNew.temperature
    )
    ctdNew.sigma.attrs['units'] = 'kg/m^3'
    ctdNew.sigma.attrs['notes'] = 'potential density anomaly (sigma0)'

    return ctdNew


def process_deployment_ctd(deployment_path,processed_path,lat,lon,overwrite=False):
    deployment_num = '_'.join(deployment_path.name.split('_')[:-1])
    proc_file = processed_path / f'dragon_{deployment_num}_ctd.nc'
    if not overwrite and proc_file.is_file():
        print(f'skipping {proc_file.name}, already created')
        output = xr.load_dataset(proc_file)
    else:
        # Skips macOS metadata files (._*) that cause error
        ctd_paths = [path for path in deployment_path.rglob('*.rsk') if not path.name.startswith('._')]
        processed_ctd_list = []
        for file in sorted(ctd_paths):
            raw_save_file = file.parent / f'dragon_ctd_{file.with_suffix(".nc").name}'
            if not overwrite and raw_save_file.is_file():
                print(f'skipping {file.name} -> {raw_save_file.name}, already converted')
                with xr.open_dataset(raw_save_file) as raw_ctd:
                    print(f'cleaning {raw_save_file.name}')
                    cleaned_ctd = initial_clean_dragon_ctd(raw_ctd,lat=lat,lon=lon)
            else:
                print(f'converting {file.name}')
                raw_ctd = readRSK(file)
                file_save(raw_ctd,raw_save_file,overwrite)
                print(f'cleaning {raw_save_file.name}')
                cleaned_ctd = initial_clean_dragon_ctd(raw_ctd,lat=lat,lon=lon)
            processed_ctd_list.append(cleaned_ctd)
        if not processed_ctd_list:
            raise ValueError(f'no processed ctd data for {proc_file}\nProbably an error in the code.')
        elif len(processed_ctd_list)>1:
            print('combining multiple ctd files into one dataset')
            output = xr.combine_by_coords(processed_ctd_list)
        else:
            output = processed_ctd_list[0]
        file_save(output, proc_file, overwrite)
        print('')  # noqa: FURB105
    return output
