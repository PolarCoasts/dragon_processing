import gsw
import numpy as np
import xarray as xr
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from collections import defaultdict

from .utilities import (
    amplitude_to_backscatter,
    dataset_convert_timestamp,
    find_nearest_indices,
    interp_datetime64,
    file_save
)


def find_sig_folders(dep_path):
    sig_paths = []
    for path in dep_path.rglob('sig*'):
        if path.is_dir():
            sig_paths.append(path)
    return sorted(sig_paths)

#region adcp coordinate systems

def define_rotation_nortek_XYZ_to_ENU(heading,roll,pitch,degrees=False,orientation='up'):
    # Defines a rotation from Nortek XYZ coordinates to East-North-Up (ENU) coordinates based on the heading, roll, and pitch of the instrument.
    # degrees is a boolean that specifies whether the angles are in degrees (True) or radians (False).
    # orientation is a string that specifies whether the instrument is programmed to be oriented 'up' or 'down'.

    if degrees:
        conversion = 1
    else:
        conversion = np.pi/180

    match orientation:
        case 'up':
            xAngle = roll
        case 'down':
            xAngle = roll + 180*conversion
        case _:
            raise ValueError('orientation invalid. must be either up or down.')

    yAngle = -pitch

    zAngle = -(heading-90*conversion)

    return Rotation.from_euler(seq='xyz',angles=[xAngle,yAngle,zAngle],degrees=degrees)

def get_beam_direction_nortek_instrument_coords(beam_angle,degrees=True,fiveBeams=False):
    # Returns the direction of the beams in Nortek instrument coordinates based on the beam angles.
    # fiveBeams is a boolean that specifies whether the instrument has 5 beams (True) or 4 beams (False).
    # note, this function allows beam_angle to be passed as an array for each of the 4 beams or as a single number.
    # the fifth beam is always assumed to be [0,0,1] (upward looking).
    # I can definitely do some improvements to make this function more robust in the future.

    if degrees:
        beam_angle = beam_angle*np.pi/180

    if np.size(beam_angle)==1:
        beam_angle = [beam_angle,beam_angle,beam_angle,beam_angle]

    beam1 = [np.sin(beam_angle[0]),0,np.cos(beam_angle[0])]
    beam2 = [0,-np.sin(beam_angle[1]),np.cos(beam_angle[1])]
    beam3 = [-np.sin(beam_angle[2]),0,np.cos(beam_angle[2])]
    beam4 = [0,np.sin(beam_angle[3]),np.cos(beam_angle[3])]

    if fiveBeams:
        beam5 = [0,0,1]
        return [beam1,beam2,beam3,beam4,beam5]
    else:
        return [beam1,beam2,beam3,beam4]

def get_beam_direction_nortek_earth_coords(beam_angle,heading,roll,pitch,degrees,orientation,fiveBeams):
    # Returns the direction of the beams in Earth coordinates based on the beam angles and the instrument orientation.
    # degrees is a boolean that specifies whether the angles are in degrees (True) or radians (False).
    # orientation is a string that specifies whether the instrument is programmed to be oriented 'up' or 'down'.
    # fiveBeams is a boolean that specifies whether the instrument has 5 beams (True) or 4 beams (False).
    # note, this function allows beam_angle to be passed as an array for each of the 4 beams or as a single number.
    # the fifth beam is always assumed to be [0,0,1] (upward looking) in instrument coords.

    beam_directions_instrument = get_beam_direction_nortek_instrument_coords(beam_angle,degrees=degrees,fiveBeams=fiveBeams)
    rotation_to_earth = define_rotation_nortek_XYZ_to_ENU(heading,roll,pitch,degrees=degrees,orientation=orientation)
    beam_directions_earth = [rotation_to_earth.apply(beamDir) for beamDir in beam_directions_instrument]
    return beam_directions_earth

def rotate_signature_XYZ_to_ENU(time,timeFull,velXYZ,heading,roll,pitch,orientation='up',degrees=True):
    # Rotates the XYZ velocities to ENU coordinates based on the heading, roll, and pitch of the instrument.
    # time is the time coordinate of the velocities.
    # timeFull is the full time coordinate of the dataset (used for interpolation).
    # velXYZ is a >2D array with shape (time, ..., 3) where 3 corresponds to X, Y, Z.
    # heading, roll, and pitch are 1D arrays with shape (timeFull,) depending on the orientation.
    # orientation is a string that specifies whether the instrument is programmed to be oriented 'up' or 'down'.
    # degrees is a boolean that specifies whether heading, roll, and pitch are in degrees (True) or radians (False).

    
    
    H = interp_datetime64(time, timeFull, heading, axis=0)
    R = interp_datetime64(time, timeFull, roll, axis=0)
    P = interp_datetime64(time, timeFull, pitch, axis=0)

    velENU = [define_rotation_nortek_XYZ_to_ENU(h,r,p,degrees=degrees,orientation=orientation).apply(np.transpose([vX,vY,vZ])) 
              for [h,r,p,vX,vY,vZ] in zip(H,R,P,velXYZ[...,0],velXYZ[...,1],velXYZ[...,2])]

    return np.array(velENU)

#region interactive functions

# def bottomDetection(amplitude, dist, edgeMask = None, edgeMin=100, edgeMax=200, amplitudeThreshold=50,maxInterpolateGap=0,edgeBlurKernelX=1, edgeBlurKernelY=1):

#     dist = utf.ensureArray(dist)
#     amplitude = utf.ensureArray(amplitude)

#     if edgeMask is None:
#         edgeMask = np.ones_like(amplitude, dtype=bool)

#     amplitudeKernel=5

#     # Convert amplitude to backscatter
#     backscatter = amplitude_to_backscatter(amplitude, dist[None, :])

#     # Detect edges using Canny edge detection
#     edges = cv2.Canny(utf.convertToUint8(cv2.GaussianBlur(backscatter, ksize=(5, 5), sigmaX=1, sigmaY=1)), edgeMin, edgeMax)

#     ampEdge = cv2.GaussianBlur(amplitude, ksize=(amplitudeKernel, amplitudeKernel), sigmaX=1, sigmaY=1)
#     ampEdge[cv2.GaussianBlur(edges,ksize=(edgeBlurKernelX,edgeBlurKernelY),sigmaX=20,sigmaY=20) == 0] = 0
#     ampEdge[ampEdge < amplitudeThreshold] = 0
#     ampEdge[~edgeMask] = 0

#     edgeIndices = np.argmax(ampEdge, axis=1)
#     edge = dist[edgeIndices]
#     edge[np.sum(ampEdge != 0, axis=1) < 1] = np.nan
#     edge[np.array([amplitude[i, edgeIndices[i]] for i in range(amplitude.shape[0])]) < amplitudeThreshold] = np.nan

#     edge = utf.nan_interp(edge, maxGap=maxInterpolateGap)

#     return edge, ampEdge

# def interactive_bottom_detection(sig, beamIndex):
#     amplitude = sig.beam_amplitude[...,beamIndex].values.copy()
#     dist = sig.z.values.copy()
#     time = sig.time.values.copy()
#     bottomDistance = sig.bottomDistance_beam[...,beamIndex].values.copy()
#     mask = sig.bottomDetectionMask[...,beamIndex].values.copy()
#     edgeMin = sig.bottomDetectionEdgeMin[beamIndex].values.item()
#     edgeMax = sig.bottomDetectionEdgeMax[beamIndex].values.item()
#     amplitudeThreshold = sig.bottomDetectionAmplitudeThreshold[beamIndex].values.item()
#     skipped = sig.bottomDetectionSkipped[beamIndex].values.item()
#     maxInterpolateGap = sig.bottomDetectionMaxInterpolateGap[beamIndex].values.item()
#     edgeBlurKernelX = sig.bottomDetectionEdgeBlurKernelX[beamIndex].values.item()
#     edgeBlurKernelY = sig.bottomDetectionEdgeBlurKernelY[beamIndex].values.item()

#     ampEdge = np.zeros_like(amplitude)

#     def updateBottomDetection():
#         nonlocal bottomDistance, ampEdge, edgeMin, edgeMax, amplitudeThreshold, maxInterpolateGap, edgeBlurKernelX, edgeBlurKernelY
#         bottomDistance, ampEdge = bottomDetection(amplitude, dist, edgeMask=mask, edgeMin=edgeMin, edgeMax=edgeMax, amplitudeThreshold=amplitudeThreshold, maxInterpolateGap=maxInterpolateGap, edgeBlurKernelX=edgeBlurKernelX, edgeBlurKernelY=edgeBlurKernelY)

#     updateBottomDetection()

#     detection_finished = False

#     timePlot = time.astype(float)
#     timePlot = (timePlot - timePlot[0])*10**-9

#     xselected = np.full(2, np.nan)
#     yselected = np.full(2, np.nan)

#     xscatter, yscatter = np.meshgrid(np.linspace(timePlot.min(), timePlot.max(), 100), np.linspace(dist.min(), dist.max(), 100))
#     xscatter = xscatter.flatten()
#     yscatter = yscatter.flatten()

#     scatter1 = go.Scatter(x=xscatter, y=yscatter, mode='markers', marker=dict(opacity=0), showlegend=False)
#     scatter2 = go.Scatter(x=xscatter, y=yscatter, mode='markers', marker=dict(opacity=0), showlegend=False)

#     heatmap1 = go.Heatmap(x=timePlot, y=dist, z=(amplitude*mask).transpose(), colorscale='amp', zmin=0, zmax=100, colorbar=dict(x=1), showlegend=False)
#     heatmap2 = go.Heatmap(x=timePlot, y=dist, z=ampEdge.transpose(), colorscale='amp', zmin=0, zmax=100, colorbar=dict(x=1), showlegend=False)

#     edgeScatter = go.Scatter(y=bottomDistance, x=timePlot, mode='lines', marker=dict(color='blue', size=5), name='distance', showlegend=True)
#     BTscatter = go.Scatter(y=sig.beam_bottom_track_distance_BT[...,beamIndex].values, x=timePlot, mode='lines', marker=dict(color='red', size=5), name='BT distance', showlegend=True)

#     # Set up widget

#     figWidget = go.FigureWidget(make_subplots(rows=2, cols=1, row_titles=['Raw Amplitude + Detected Bottom', 'processed amplitude for bottom detection']))
#     figWidget.update_layout(height=600, width=1000,xaxis2=dict(matches='x1',title='distance from ADCP (m)'),yaxis2=dict(matches='y1'),yaxis1=dict(title='time (s)'),xaxis1=dict(title='distance from ADCP (m)'))
#     figWidget.add_traces([scatter1,scatter2,heatmap1,heatmap2,edgeScatter,BTscatter], rows=[1, 2, 1, 2, 1, 1], cols=[1, 1, 1, 1, 1, 1])

#     descriptionWidth = 200
#     widgetWidth = 600
    
#     # Create widgets
#     edgeMin_widget = widgets.IntSlider(value=edgeMin, min=0, max=255, step=1, description='edgeMin',layout=widgets.Layout(width=f'{widgetWidth}px'),style={'description_width': f'{descriptionWidth}px'})
#     edgeMax_widget = widgets.IntSlider(value=edgeMax, min=0, max=255, step=1, description='edgeMax',layout=widgets.Layout(width=f'{widgetWidth}px'),style={'description_width': f'{descriptionWidth}px'})
#     amplitudeThreshold_widget = widgets.IntSlider(value=amplitudeThreshold, min=0, max=100, step=1, description='amplitudeThreshold',layout=widgets.Layout(width=f'{widgetWidth}px'),style={'description_width': f'{descriptionWidth}px'})
#     maxInterpolateGap_widget = widgets.IntSlider(value=maxInterpolateGap, min=0, max=100, step=1, description='maxInterpolateGap',layout=widgets.Layout(width=f'{widgetWidth}px'),style={'description_width': f'{descriptionWidth}px'})
#     edgeBlurKernelX_widget = widgets.IntSlider(value=edgeBlurKernelX, min=1, max=101, step=2, description='edgeBlurKernelX',layout=widgets.Layout(width=f'{widgetWidth}px'),style={'description_width': f'{descriptionWidth}px'})
#     edgeBlurKernelY_widget = widgets.IntSlider(value=edgeBlurKernelY, min=1, max=101, step=2, description='edgeBlurKernelY',layout=widgets.Layout(width=f'{widgetWidth}px'),style={'description_width': f'{descriptionWidth}px'})

#     reset1_button = widgets.Button(description="Reset Mask")
#     reset2_button = widgets.Button(description="Return To Starting Values")
#     update_button = widgets.Button(description="Update Plot")
#     select_button = widgets.Button(description="Select Region")
#     submitSelection_button = widgets.Button(description="Submit Selection",disabled=True)
#     cancelSelection_button = widgets.Button(description="Cancel Selection",disabled=True)
#     finish_button = widgets.Button(description="Finish Bottom Detection")
#     skip_button = widgets.Button(description="Skip Bottom Detection", button_style='warning')

#     button_box = widgets.HBox([reset1_button, reset2_button, update_button, select_button, submitSelection_button, cancelSelection_button, finish_button, skip_button])

#     output = widgets.Output()
#     output2 = widgets.Output()

#     def updateValuesFromWidgets():
#         nonlocal edgeMin, edgeMax, amplitudeThreshold, maxInterpolateGap, edgeBlurKernelX, edgeBlurKernelY
#         edgeMin = edgeMin_widget.value
#         edgeMax = edgeMax_widget.value
#         amplitudeThreshold = amplitudeThreshold_widget.value
#         maxInterpolateGap = maxInterpolateGap_widget.value
#         edgeBlurKernelX = edgeBlurKernelX_widget.value
#         edgeBlurKernelY = edgeBlurKernelY_widget.value

#     def disable_buttons():
#         submitSelection_button.disabled = True
#         cancelSelection_button.disabled = True
#         finish_button.disabled = True
#         reset1_button.disabled = True
#         reset2_button.disabled = True
#         update_button.disabled = True
#         select_button.disabled = True
#         skip_button.disabled = True

#         edgeMin_widget.disabled = True
#         edgeMax_widget.disabled = True
#         amplitudeThreshold_widget.disabled = True
#         maxInterpolateGap_widget.disabled = True
#         edgeBlurKernelX_widget.disabled = True
#         edgeBlurKernelY_widget.disabled = True

#     def reset_buttons():
#         submitSelection_button.disabled = True
#         cancelSelection_button.disabled = True
#         finish_button.disabled = False
#         reset1_button.disabled = False
#         reset2_button.disabled = False
#         update_button.disabled = False
#         select_button.disabled = False
#         skip_button.disabled = False

#         edgeMin_widget.disabled = False
#         edgeMax_widget.disabled = False
#         amplitudeThreshold_widget.disabled = False
#         maxInterpolateGap_widget.disabled = False
#         edgeBlurKernelX_widget.disabled = False
#         edgeBlurKernelY_widget.disabled = False

#     def confirm_choices(message):
#         disable_buttons()

#         continueBool = False
#         confirmed = False

#         def on_confirm_clicked(c):
#             nonlocal continueBool,confirmed
#             continueBool = True
#             confirmed=True

#         def on_cancel_clicked(c):
#             nonlocal continueBool,confirmed
#             continueBool = True
#             confirmed=False

#         with output2:
#             clear_output(wait=True)
#             print(message)
#             print("Please confirm")
#             confirm_button = widgets.Button(description="Confirm", button_style='success')
#             cancel_button = widgets.Button(description="Cancel", button_style='danger')
#             confirm_button.on_click(on_confirm_clicked)
#             cancel_button.on_click(on_cancel_clicked)
#             button_box = widgets.HBox([confirm_button, cancel_button])
#             display(button_box)

#         with ui_events() as poll2:
#             while not continueBool:
#                 poll2(10)
#                 time_library.sleep(1)
#         return confirmed
    
#     def selection_callback(trace, points, selector):
#         nonlocal xselected, yselected
#         # Mask selected points (set to False)
#         if selector and isinstance(selector.xrange, Iterable) and isinstance(selector.yrange, Iterable):
#             xselected = [float(x) for x in selector.xrange]
#             yselected = [float(temp) for temp in selector.yrange]

#             with output2:
#                 clear_output(wait=True)
#                 print(f"Selected range: x={xselected}, y={yselected}")

#         submitSelection_button.disabled = False

#     def start_selection(b=None):
#         nonlocal figWidget
#         # Enable selection mode for both figures

#         figWidget.data[0].on_selection(selection_callback)
#         figWidget.data[1].on_selection(selection_callback)

#         figWidget.layout.dragmode = 'select'
#         disable_buttons()
#         cancelSelection_button.disabled = False

#     def finish_selection(b=None):
#         nonlocal xselected, yselected

#         for i, xi in enumerate(timePlot):
#             for j, yj in enumerate(dist):
#                 if xselected[0] <= xi <= xselected[1] and yselected[0] <= yj <= yselected[1]:
#                     mask[i, j] = False

#         reset_buttons()

#         figWidget.data[0].on_selection(None)
#         figWidget.data[1].on_selection(None)

#         for trace in figWidget.data:
#             if hasattr(trace,'selectedpoints'):
#                 trace.selectedpoints = []

#         figWidget.layout.dragmode = 'zoom'

#         update_plot()
#         with output2:
#             clear_output(wait=True)
#             print(f"Mask updated for selected range: x={xselected}, y={yselected}")

#     def cancel_selection(b=None):
#         reset_buttons()
#         figWidget.data[0].on_selection(None)
#         figWidget.data[1].on_selection(None)
        
#         for trace in figWidget.data:
#             if hasattr(trace,'selectedpoints'):
#                 trace.selectedpoints = []
        
#         figWidget.layout.dragmode = 'zoom'
        
#         with output2:
#             clear_output(wait=True)
#             print("Selection cancelled. You can continue editing.")

#     def update_plot(b=None):
#         nonlocal figWidget
#         nonlocal bottomDistance, ampEdge, mask

#         updateValuesFromWidgets()

#         updateBottomDetection()

#         with figWidget.batch_update():
#             figWidget.data[2].z = (amplitude * mask).transpose()
#             figWidget.data[3].z = ampEdge.transpose()
#             figWidget.data[4].y = bottomDistance

#     def reset1_callback(b=None):
#         nonlocal mask
#         mask = np.ones_like(amplitude, dtype=bool)
#         update_plot()

#     def reset2_callback(b=None):
#         nonlocal mask
#         mask = sig.bottomDetectionMask[...,beamIndex].values.copy()
#         edgeMin_widget.value = sig.bottomDetectionEdgeMin[beamIndex].values.item()
#         edgeMax_widget.value = sig.bottomDetectionEdgeMax[beamIndex].values.item()
#         amplitudeThreshold_widget.value = sig.bottomDetectionAmplitudeThreshold[beamIndex].values.item()
#         maxInterpolateGap_widget.value = sig.bottomDetectionMaxInterpolateGap[beamIndex].values.item()
#         edgeBlurKernelX_widget.value = sig.bottomDetectionEdgeBlurKernelX[beamIndex].values.item()
#         edgeBlurKernelY_widget.value = sig.bottomDetectionEdgeBlurKernelY[beamIndex].values.item()
#         update_plot()

#     def finish_bottom_detection(b=None):
#         update_plot()
#         nonlocal detection_finished, skipped, edgeMin,edgeMax,amplitudeThreshold

#         confirmed = confirm_choices(f"Finishing bottom detection with results shown in plot above \n" \
#         "Edge Min: {edgeMin}, Edge Max: {edgeMax}, Amplitude Threshold: {amplitudeThreshold} \n" \
#         "Please confirm or cancel to continue editing.")

#         if confirmed:
#             with output2:
#                 clear_output(wait=True)
#                 print('bottom detection results returned')
#             skipped = False
#             detection_finished = True
#         else:
#             with output2:
#                 clear_output(wait=True)
#                 print("Submission canceled. You can continue editing.")
#                 reset_buttons()

#     def skip_bottom_detection(b=None):
#         nonlocal detection_finished, bottomDistance, ampEdge, mask, edgeMin, edgeMax, amplitudeThreshold, maxInterpolateGap, skipped

#         confirmed = confirm_choices("Skipping bottom detection and returning starting values")

#         if confirmed:
#             bottomDistance = sig.bottomDistance_beam[...,beamIndex].values
#             mask = sig.bottomDetectionMask[...,beamIndex].values
#             edgeMin = sig.bottomDetectionEdgeMin[beamIndex].values
#             edgeMax = sig.bottomDetectionEdgeMax[beamIndex].values
#             amplitudeThreshold = sig.bottomDetectionAmplitudeThreshold[beamIndex].values
#             maxInterpolateGap = sig.bottomDetectionMaxInterpolateGap[beamIndex].values
#             edgeBlurKernelX = sig.bottomDetectionEdgeBlurKernelX[beamIndex].values
#             edgeBlurKernelY = sig.bottomDetectionEdgeBlurKernelY[beamIndex].values
#             skipped = True
#             detection_finished = True
#             with output2:
#                 clear_output(wait=True)
#                 print("Bottom detection skipped.")
#         else:
#             reset_buttons()
#             with output2:
#                 clear_output(wait=True)
#                 print("Skip cancelled. You can continue editing.")

#     reset1_button.on_click(reset1_callback)
#     reset2_button.on_click(reset2_callback)
#     update_button.on_click(update_plot)
#     select_button.on_click(start_selection)
#     submitSelection_button.on_click(finish_selection)
#     cancelSelection_button.on_click(cancel_selection)
#     finish_button.on_click(finish_bottom_detection)
#     skip_button.on_click(skip_bottom_detection)

#     display(widgets.VBox([edgeMin_widget, edgeMax_widget, amplitudeThreshold_widget, maxInterpolateGap_widget, edgeBlurKernelX_widget, edgeBlurKernelY_widget, button_box, output,output2]))
#     with output:
#         clear_output(wait=True)
#         display(figWidget)

#     with ui_events() as poll:
#         while not detection_finished:
#             poll(10)
#             time_library.sleep(1)

#     if not skipped:
#         sig.bottomDistance_beam[...,beamIndex] = bottomDistance
#         sig.bottomDetectionMask[...,beamIndex] = mask
#         sig.bottomDetectionEdgeMin[beamIndex] = edgeMin
#         sig.bottomDetectionEdgeMax[beamIndex] = edgeMax
#         sig.bottomDetectionAmplitudeThreshold[beamIndex] = amplitudeThreshold
#         sig.bottomDetectionSkipped[beamIndex] = False
#         sig.bottomDetectionMaxInterpolateGap[beamIndex] = maxInterpolateGap
#         sig.bottomDetectionEdgeBlurKernelX[beamIndex] = edgeBlurKernelX
#         sig.bottomDetectionEdgeBlurKernelY[beamIndex] = edgeBlurKernelY

#     return sig


#region signature processing

def signature_transpose(sig):
    coordOrder = ['time' if mode == 'normal' else 'time_' + mode for mode in sig.modes] + ['z', 'beam', 'direction']
    return sig.transpose(*coordOrder)

def create_signature_coordinates(sigProc,sigRaw,sig_config):
    # Creates the coordinates for the processed Signature dataset based on the raw dataset and sig_config file.
    modes = sigRaw.modes

    # for each mode specify a dictionary of the coordinate variables to create with the processed names as the key and a tuple of the raw name, and a dictionary of extra attributes as the value.
    mode_coords = {
        'normal': {
            'time': ('Time',
                     {'notes': 'timestamp for standard measurements',
                     'sampling_rate': str(sig_config.burst_sampleRate.data) + ' ' + str(sig_config.burst_sampleRate.units)}),
            'z': ('Z',
                  {'units': str(sig_config.burst_cellSize.units),
                  'notes': 'distance from signature for standard measurements',
                  'cellHeight': str(sig_config.burst_cellSize.data) + ' ' + str(sig_config.burst_cellSize.units)}),
                  },
        'BT': {
            'time_BT': ('BTTime',
                         {'notes': 'timestamp for bottom track measurements',
                          'sampling_rate': str(sig_config.burst_sampleRate.data) + ' ' + str(sig_config.burst_sampleRate.units)}),
        },
        'I': {
            'time_I': ('ITime',
                       {'notes': 'timestamp for beam 5 measurements',
                        'sampling_rate': str(sig_config.burst_sampleRate.data) + ' ' + str(sig_config.burst_sampleRate.units)}),
        }
    }

    for mode in modes:
        if mode not in mode_coords:
            raise ValueError(f'mode {mode} not yet supported. currently supporting normal, I, and BT.')
        else:
            for newName, (rawName, attrs) in mode_coords[mode].items():
                sigProc.coords[newName] = sigRaw[rawName].data
                sigProc[newName].attrs = sigRaw[rawName].attrs.copy()
                sigProc[newName].attrs.update(attrs)

    if 'I' in modes:
        sigProc.coords['beam'] = np.arange(1,6)
    else:
        sigProc.coords['beam'] = np.arange(1,5)

    sigProc.coords['XYZ'] = ('direction', ['X','Y','Z'])
    sigProc.coords['ENU'] = ('direction', ['East','North','Up'])

    return sigProc

def load_signature_mode_specific(sigProc, sigRaw, mode):

    """
    Extracts and renames mode-specific variables from a raw Signature ADCP xarray dataset (sigRaw)
    and populates a new, cleaned dataset (sigProc) for the specified mode, using the provided config.
    Ensures variables are assigned with the correct coordinates.
    """

    # variables that will be grouped in the beam coordinate
    # key is the name of the raw variable, value is a tuple of the new name and the coordinates to assign variable to in dataset
    beam_vars = {
        'CorBeam':('beam_correlation', ('beam','time','z')),
        'AmpBeam':('beam_amplitude', ('beam','time','z')),
        'VelBeam':('beam_velocity_l0', ('beam','time','z')),
        'FOMBeam':('beam_bottom_track_figure_of_merit', ('beam','time','z')),
        'DistanceBeam':('beam_bottom_track_distance', ('beam','time','z'))
        }
    
    xyz_vars = {
        'Accelerometer': ('acceleration_xyz', ('direction','time')),
        'Magnetometer': ('magnetometer_xyz', ('direction','time')),
        'AHRSGyro': ('gyro_xyz', ('direction','time')),
    }

    # Base variable definitions
    base_vars = {
        'WaterTemperature': ('temperature_raw', ('time',)),
        'Pressure': ('pressure', ('time',)),

        'Heading': ('heading_angle', ('time',)),
        'Pitch': ('pitch_angle', ('time',)),
        'Roll': ('roll_angle', ('time',)),
    }

    # For each mode, specify extra information.
    # Tuple contains mode-specific prefixes and time and z coordinate names
    # dictionary specifies variable names to include and extra attributes to add

    mode_attrs = {
        'normal':(('', 'time', 'z'),{

            #beam variables
            'CorBeam': {'notes': 'correlation of beam signal'},
            'AmpBeam': {'notes': 'amplitude of beam signal'},
            'VelBeam': {'notes': 'raw velocity measured by each beam.'},

            #XYZ variables
            'Accelerometer' : {'notes': 'accelerometer measurement in X, Y, and Z directions of instrument'},
            'Magnetometer' : {'notes': 'magnetometer measurement in X, Y, and Z directions of instrument'},
            'AHRSGyro' : {'notes': 'AHRS gyroscope measurement in X, Y, and Z directions of instrument'},

            #base variables
            'WaterTemperature': {'notes': 'in-situ temperature'},
            'Pressure': {'notes': 'Sea pressure corrected for air pressure'},

            'Heading': {'notes': 'heading in degrees from true north'},
            'Pitch': {'notes': 'pitch in degrees'},
            'Roll': {'notes': 'roll in degrees'}
        }),

        'BT': (('BT', 'time_BT', None),{

            # beam variables
            'VelBeam': {'notes': 'raw Bottom Track Velocity measured by each beam.'},
            'FOMBeam': {'notes': 'Bottom Track figure of merit (larger is worse) of each beam signal'},
            'DistanceBeam': {'notes': 'distance to ice-face measured by each beam'},

            # XYZ variables
            'Accelerometer': {'notes': 'accelerometer measurement in X, Y, and Z directions of instrument'},
            'Magnetometer': {'notes': 'magnetometer measurement in X, Y, and Z directions of instrument'},
            'AHRSGyro': {'notes': 'AHRS gyroscope measurement in X, Y, and Z directions of instrument'},
        
            # base variables
            'Pressure': {'notes': 'Sea pressure corrected for air pressure, for Bottom track timestamp'},
            'WaterTemperature': {'notes': 'in-situ temperature for Bottom track timestamp'},

            'Heading': {'notes': 'heading in degrees from true north for Bottom Track timestamp'},
            'Pitch': {'notes': 'pitch in degrees for Bottom Track timestamp'},
            'Roll': {'notes': 'roll in degrees for Bottom Track timestamp'}
        }),

        'I': (('I', 'time_I', 'z'),{

            # beam variables
            'VelBeam': {'notes': 'Velocity measured by beam 5'},
            'CorBeam': {'notes': 'Correlation of beam 5 signal'},
            'AmpBeam': {'notes': 'Amplitude of beam 5 signal'},

            # XYZ variables
            'Accelerometer': {'notes': 'accelerometer measurement in X, Y, and Z directions of instrument'},
            'Magnetometer': {'notes': 'magnetometer measurement in X, Y, and Z directions of instrument'},
            'AHRSGyro': {'notes': 'AHRS gyroscope measurement in X, Y, and Z directions of instrument'},

            # base variables
            'Pressure': {'notes': 'Sea pressure corrected for air pressure, for 5th beam timestamp'},
            'WaterTemperature': {'notes': 'in-situ temperature for 5th beam timestamp'},

            'Heading': {'notes': 'heading in degrees from true north for 5th beam timestamp'},
            'Pitch': {'notes': 'pitch in degrees for 5th beam timestamp'},
            'Roll': {'notes': 'roll in degrees for 5th beam timestamp'}
        }),
    }

    # Check if the mode is recognized
    if mode not in mode_attrs:
        raise ValueError(f"Mode '{mode}' not recognized.")
    
    prefix, time_coord, z_coord = mode_attrs[mode][0]
    
    for base_name_raw in mode_attrs[mode][1]:

        extra_attrs = mode_attrs[mode][1][base_name_raw] # Get extra attributes

        if base_name_raw in base_vars:
            base_name_new, base_dims = base_vars[base_name_raw] # Get raw name and dimensions
            dims = tuple(d for d in [time_coord if d == 'time' else z_coord if d == 'z' else d for d in base_dims] if d is not None) # Adjust coordinate names for this mode

            new_name = base_name_new + '_' + prefix if prefix else base_name_new
            raw_name = prefix + '_' + base_name_raw if prefix else base_name_raw

            sigProc[new_name] = (dims, sigRaw[raw_name].data)
            sigProc[new_name].attrs = sigRaw[raw_name].attrs.copy()
            sigProc[new_name].attrs.update(extra_attrs)

        elif base_name_raw in xyz_vars:
            base_name_new, base_dims = xyz_vars[base_name_raw] # Get raw name and dimensions
            dims = tuple(d for d in [time_coord if d == 'time' else z_coord if d == 'z' else d for d in base_dims] if d is not None) # Adjust coordinate names for this mode

            new_name = base_name_new + '_' + prefix if prefix else base_name_new
            raw_names = [prefix + '_' + base_name_raw + axis if prefix else base_name_raw + axis for axis in ['X','Y','Z']]

            sigProc[new_name] = (dims, [sigRaw[raw_name].data for raw_name in raw_names])
            sigProc[new_name].attrs = sigRaw[raw_names[0]].attrs.copy()
            sigProc[new_name].attrs.update(extra_attrs)

        elif base_name_raw in beam_vars:
            base_name_new, base_dims = beam_vars[base_name_raw] # Get raw name and dimensions
            dims = tuple(d for d in [time_coord if d == 'time' else z_coord if d == 'z' else d for d in base_dims] if d is not None) # Adjust coordinate names for this mode

            new_name = base_name_new + '_' + prefix if prefix else base_name_new

            if mode == 'I':
                raw_names = [None,]*4 + [prefix + '_' + base_name_raw + '5' if prefix else base_name_raw + '5']
            elif mode in ['normal','BT']:
                raw_names = [prefix + '_' + base_name_raw + str(i+1) if prefix else base_name_raw + str(i+1) for i in range(4)] + [None,]
            else:
                raise ValueError(f"Mode '{mode}' does not support beam variables.")
            
            emptyArray = np.full(tuple(sigProc[dim].size for dim in dims if dim != 'beam'), np.nan)
            sigProc[new_name] = (dims, [sigRaw[raw_name].data if raw_name is not None else emptyArray for raw_name in raw_names])
            sigProc[new_name].attrs = sigRaw[next(a for a in raw_names if a is not None)].attrs.copy()
            sigProc[new_name].attrs.update(extra_attrs)

        else:
            raise ValueError(f"Base name '{base_name_new}' not recognized in base_vars, xyz_vars, or beam_vars.")

    return sigProc

def combine_signature_full_time_data(sig):
    """
    For variables that are measured during all modes, creates a combined timeFull coordinate and stores the data
    together sorted onto that timeFull coordinate.
    """

    modes = sig.modes

    vars = ['temperature_raw','pressure', 
            'acceleration_xyz', 'magnetometer_xyz', 'gyro_xyz',
            'heading_angle', 'pitch_angle', 'roll_angle']
    

    timeArray = [sig.time.data if mode == 'normal' else sig['time_'+mode].data for mode in modes]
    timeFull = np.concatenate(timeArray)
    
    iSort = np.argsort(timeFull)

    sig.coords['time_full'] = timeFull[iSort]
    sig.time_full.attrs = sig.time.attrs.copy()
    sig.time_full.attrs['notes'] = 'Combined time coordinate for all modes'

    for var in vars:
        attributes = sig[var].attrs.copy()
        dataArray = [sig[var].rename({'time':'time_full'}) if mode == 'normal' else sig[var+'_'+mode].rename({'time_'+mode:'time_full'}) for mode in modes]
        sig[var] = xr.concat(dataArray,dim='time_full').isel(time_full=iSort)
        sig[var].attrs = attributes
        #drop variables for individual modes
        sig = sig.drop_vars([var+'_'+mode for mode in modes if mode != 'normal'])

    return sig

def pressure_correction_signature(sig,lat):

    pressure = sig['pressure'].data.copy()
    pressure = pressure-np.mean(pressure[pressure<np.min(pressure)+1])
    depth = np.abs(gsw.conversions.z_from_p(pressure, lat))

    sig['pressure'].data = pressure
    sig['depth'] = ('time_full', depth)
    sig['depth'].attrs['units'] = 'meters'
    sig['depth'].attrs['notes'] = 'calculated from pressure and latitude using gsw python package'

    return sig

def reformat_signature_beam_5_data(sig, beam5variable=None):
    if 'I' not in sig.modes:
        raise ValueError("Signature dataset does not contain 'I' mode data.")
    
    vars = ['beam_velocity_l0','beam_correlation','beam_amplitude']

    idx = find_nearest_indices(sig.time_I.data, sig.time.data)
    # if sig.time.size > sig.time_I.size:
    _,idx_first = np.unique(idx,return_index=True)
    idxNaN = np.ones(idx.shape,dtype=bool)
    idxNaN[idx_first] = False
    # else:
    #     idxNaN = np.zeros(idx.shape,dtype=bool)

    time = sig.time_I.data[idx]
    time[idxNaN] = np.datetime64('NaT')
    attrs = sig.time_I.attrs.copy()

    for var in vars:
        data = sig[var+'_I'].data[idx,:,4].copy()
        data[idxNaN,:] = np.nan
        sig[var].data[:,:,4] = data #copy beam 5 data to the 5th beam location
        sig = sig.drop_vars([var+'_I']) #drop the beam 5 variable

    if beam5variable is not None:
        newB5var = beam5variable[idx,...]
        newB5var[idxNaN,...] = np.nan

    sig = sig.drop_dims('time_I') #drop the time_I coordinate

    sig.coords['time_I'] = ('time', time)
    sig.time_I.attrs = attrs
    sig.time_I.attrs['notes'] = 'timestamp for beam 5 measurements that are being treated as on the same time-stamp as the other beams for velocity calculations'

    if beam5variable is not None:
        return sig, newB5var
    else: 
        return sig

def reformat_signature_bottom_track_data(sig, BTvariable=None):
    if 'BT' not in sig.modes:
        raise ValueError("Signature dataset does not contain 'BT' mode data.")

    vars = ['beam_velocity_l0_BT','beam_bottom_track_figure_of_merit_BT','beam_bottom_track_distance_BT']

    idx = find_nearest_indices(sig.time_BT.data, sig.time.data)
    # if sig.time.data.size > sig.time_BT.data.size:
    _,idx_first = np.unique(idx, return_index=True)
    idxNaN = np.ones(idx.size,dtype=bool)
    idxNaN[idx_first] = False
    # else:
    #     idxNaN = np.zeros(idx.size,dtype=bool)

    time_BT = sig.time_BT.data[idx]
    time_BT[idxNaN] = np.datetime64('NaT')
    time_attrs = sig.time_BT.attrs.copy()

    for var in vars:
        data = sig[var].data.copy()[idx,...]
        data[idxNaN,...] = np.nan
        attrs = sig[var].attrs.copy()
        dims = tuple('time' if d == 'time_BT' else d for d in sig[var].dims)

        sig.drop_vars(var)
        sig[var] = (dims,data)
        sig[var].attrs = attrs

    if BTvariable is not None:
        newBTvar = BTvariable[idx,...]
        newBTvar[idxNaN,...] = np.nan

    sig = sig.drop_dims('time_BT')
    sig.coords['time_BT'] = ('time',time_BT)
    sig.time_BT.attrs = time_attrs
    
    if BTvariable is not None:
        return sig, newBTvar
    else: 
        return sig

def clean_signature_beam_velocities(sig, error, beam5=False, error5=None, BT=False, errorBT=None):

    assert np.unique(sig.time_full).size == sig.time_full.size, "time_full coordinate contains duplicate values"

    # Get QA/QC constants
    roll_cutoff = sig.roll_cutoff
    pitch_cutoff = sig.pitch_cutoff
    cor_cutoff = sig.correlation_cutoff
    amp_cutoff = sig.amplitude_cutoff

    # velBeam: (time, z, beam)
    cor = sig['beam_correlation'].values
    amp = sig['beam_amplitude'].values

    # Mask bad correlations and amplitudes for all beams
    mask_qc_beam_velocity = (cor < cor_cutoff) | (amp < amp_cutoff)

    # Mask for bad roll/pitch or error code
    time_full_index_normal = np.isin(sig.time_full, sig.time)
    mask_qc_beam_velocity[(sig.roll_angle[time_full_index_normal] > roll_cutoff) | (sig.pitch_angle[time_full_index_normal] > pitch_cutoff) | (error != 0),:,:4] = True

    if beam5:
        if 'I' not in sig.modes:
            raise ValueError("Signature dataset does not contain 'I' mode data, but beam5=True was specified.")
        
        nat_mask = np.isnat(sig.time_I.values)
        if np.any(nat_mask):
            mask_qc_beam_velocity[nat_mask,:,:] = True

        time_full_index_5th_beam = np.isin(sig.time_full, sig.time_I)
        mask_qc_beam_velocity[np.where(~nat_mask)[0][(sig.roll_angle[time_full_index_5th_beam] > roll_cutoff) | (sig.pitch_angle[time_full_index_5th_beam] > pitch_cutoff) | (error[~nat_mask] != 0)],:,4] = True

    # Save back to dataset
    sig['mask_qc_beam_velocity'] = (('time','z','beam'),mask_qc_beam_velocity)
    sig.mask_qc_beam_velocity.attrs['notes'] = 'True where velocity has low correlation/amplitude or bad roll/pitch/error code'

    if BT:
        if 'BT' not in sig.modes:
            raise ValueError("Signature dataset does not contain 'BT' mode data, but BT=True was specified.")
        
        assert 'time_BT' in sig.coords, "time_BT coordinate not found in dataset"
        assert 'time_BT' not in sig.dims, "time_BT should not be a dimension in the dataset"
        assert 'time' in sig.time_BT.dims, "time_BT coordinate is not linked to time dimension"

        bottom_track_fom_cutoff = sig.bottom_track_fom_cutoff

        fom = sig.beam_bottom_track_figure_of_merit_BT.values

        mask_qc_beam_velocity_BT = fom > bottom_track_fom_cutoff #don't use bottom track velocities where figure of merit is above cutoff

        #don't use bottom track velocities where time_BT is NaT
        nat_mask = np.isnat(sig.time_BT.values)
        if np.any(nat_mask):
            mask_qc_beam_velocity_BT[nat_mask,:] = True

        # Mask for bad roll/pitch or error code
        time_full_index_BT = np.isin(sig.time_full, sig.time_BT)
        mask_qc_beam_velocity_BT[np.where(~nat_mask)[0][(sig.roll_angle[time_full_index_BT] > roll_cutoff) | (sig.pitch_angle[time_full_index_BT] > pitch_cutoff) | (errorBT[~nat_mask] != 0)],:4] = True

        # if we only have good BT for a single time step, assume it is bad and mask it, since we are getting a few individual velocities that are looking clearly wrong.
        mask_qc_beam_velocity_BT[1:-1,:] = mask_qc_beam_velocity_BT[1:-1,:] | (mask_qc_beam_velocity_BT[:-2,:].astype(int) + mask_qc_beam_velocity_BT[2:,:].astype(int) == 2)

        # Save back to dataset
        sig['mask_qc_beam_velocity_BT'] = (('time','beam'),mask_qc_beam_velocity_BT)
        sig.mask_qc_beam_velocity_BT.attrs['notes'] = 'True where BT velocity has low figure of merit or bad roll/pitch/error code or where there is only a single good BT time step surrounded by bad time steps'

    return sig

def calc_signature_beam_directions_and_distance_to_surface(sig):
        
    sig['beam_directions_earth'] = (('time_full','beam','direction'),np.array([get_beam_direction_nortek_earth_coords(sig.beam_angle,heading,roll,pitch,degrees=True,orientation='up',fiveBeams='I' in sig.modes) for [heading,roll,pitch] in zip(sig.heading_angle.values,sig.roll_angle.values,sig.pitch_angle.values)]))
    sig.beam_directions_earth.attrs['notes'] = 'Beam directions in Earth coordinates. Calculated from the beam angle, heading, roll, and pitch of the instrument.'

    sig['beam_distance'] = (('z','beam'),np.array([sig.z.data.copy()/np.cos(sig.beam_angle*np.pi/180) if i<4 else sig.z.data.copy() for i in range(5)]).transpose())
    sig.beam_distance.attrs['units'] = sig.z.units
    sig.beam_distance.attrs['reference'] = sig.z.reference
    sig.beam_distance.attrs['notes'] = 'distance along beam of each bin for each beam'


    sig['bottom_distance_along_beam'] = sig.beam_bottom_track_distance_BT.copy()/np.array([np.cos(sig.beam_angle*np.pi/180) if i<4 else 1 for i in range(5)]).transpose()
    sig.bottom_distance_along_beam.attrs['notes'] = 'distance in the along beam direction'

    adcpDepth = np.repeat(np.repeat(dataset_convert_timestamp(sig,'depth',time_from='time_full',time_to='time')[:,np.newaxis,np.newaxis],sig.z.size,axis=1),sig.beam.size,axis=2)
    adcpDepth_BT = np.repeat(dataset_convert_timestamp(sig,'depth',time_from='time_full',time_to='time_BT')[:,np.newaxis],sig.beam.size,axis=1)

    beam_distance = np.repeat(sig.beam_distance.data.copy()[np.newaxis,:,:],sig.time.size,axis=0)
    bottom_distance = sig.bottom_distance_along_beam.data.copy()

    beam_directions = np.repeat(
        dataset_convert_timestamp(sig,'beam_directions_earth',time_from='time_full',time_to='time')[:,np.newaxis,:,2],
        sig.z.size, axis=1)

    beam_directions_BT = dataset_convert_timestamp(sig,'beam_directions_earth',time_from='time_full',time_to='time_BT')[:,:,2]

    sig['beam_depth'] = (('time','z','beam'), -beam_distance*beam_directions + adcpDepth)
    sig.beam_depth.attrs['notes'] = 'depth of each beam bin for each timestep and beam, using adcp pressure sensor depth and orientation with beam angle'
    sig['bottom_track_depth'] = (('time','beam'), -bottom_distance*beam_directions_BT + adcpDepth_BT)
    sig.bottom_track_depth.attrs['notes'] = 'depth of wall as seen by bottom track of each beam for each timestep using adcp pressure sensor depth and orientation with beam angle'

    return sig

def calc_backscatter(sig):
    # Calculate backscatter from amplitude and distance along beam, and save as new variable in dataset.

    if 'beam_amplitude' not in sig.data_vars or 'beam_distance' not in sig.data_vars:
        raise ValueError("Dataset must contain 'beam_amplitude' and 'beam_distance' variables to calculate backscatter.")
    
    amp = sig.beam_amplitude.values
    distance = sig.beam_distance.values[np.newaxis,:,:].repeat(sig.time.size,axis=0)
    backscatter = amplitude_to_backscatter(amp, distance)
    sig['beam_backscatter'] = (('time','z','beam'), backscatter)
    sig.beam_backscatter.attrs['units'] = 'relative backscatter'
    sig.beam_backscatter.attrs['notes'] = 'Relative backscatter calculated from amplitude and distance along beam following  backscatter = 10*log10(amp) + 2 * 9.793e-3 * distance + 20 * log10(distance). This is approximate and only useful for looking at relative differences in backscatter, not absolute values.'

    return sig

def three_beam_mask_calc(vel):
    threem_beam_mask = np.zeros(vel.shape, dtype=bool)
    # 3-beam solution: fill missing beams where only one is missing
    nan_mask = np.isnan(vel)
    for b in range(4):  # Only for beams 0-3 (assuming 4-beam transformation)
        other_beams = [i for i in range(4) if i != b]
        only_b_nan = nan_mask[...,b] & ~np.any(nan_mask[...,other_beams], axis=-1)

        threem_beam_mask[only_b_nan,b] = True  # note which beam was filled in mask

    return threem_beam_mask

def three_beam_calc(velIn,mask,convert):
    vel = velIn.copy()
    for b in range(4):
        ind = np.where(mask[...,b])[0]
        if len(ind)>0:
            if b == 0:
                vel[ind,0] = (vel[ind,1] * convert[3,1] + vel[ind,3] * convert[3,3] - vel[ind,2] * convert[2,2]) / convert[2,0]
            if b == 1:
                vel[ind,1] = (vel[ind,0] * convert[2,0] + vel[ind,2] * convert[2,2] - vel[ind,3] * convert[3,3]) / convert[3,1]
            if b == 2:
                vel[ind,2] = (vel[ind,1] * convert[3,1] + vel[ind,3] * convert[3,3] - vel[ind,0] * convert[2,0]) / convert[2,2]
            if b == 3:
                vel[ind,3] = (vel[ind,0] * convert[2,0] + vel[ind,2] * convert[2,2] - vel[ind,1] * convert[3,1]) / convert[3,3]

    return vel

def convert_to_XYZ_4beam(velBeam,convert,errVelCutoff):
    # Converts the beam velocities to XYZ coordinates using the conversion matrix.
    # velBeam is a 3D array with shape (time, depth, n) where n is at least 4 for the 4 main adcp beams.
    # convert is a 4x4 conversion matrix.
    vX = velBeam[...,0]*convert[0,0] + velBeam[...,2]*convert[0,2]
    vY = velBeam[...,1]*convert[1,1] + velBeam[...,3]*convert[1,3]
    vZ1 = velBeam[...,0]*convert[2,0] + velBeam[...,2]*convert[2,2]
    vZ2 = velBeam[...,1]*convert[3,1] + velBeam[...,3]*convert[3,3]
    vZ = (vZ1+vZ2)/2

    vError = np.abs(vZ1-vZ2)
    loc = vError > errVelCutoff
    vX[loc] = np.nan
    vY[loc] = np.nan
    vZ[loc] = np.nan

    return vX, vY, vZ, vError

def signature_velocity_correction(sig):

    #conversion matrix for converting between beam and XYZ coordinates for the 4 main beams based on instrument configuration
    convert = sig.beam_to_XYZ_conversion.reshape(4,4)
    
    #get raw beam velocities and apply QC mask to set bad velocities to NaN
    vRef = sig.beam_velocity_l0.data.copy()
    vRef[sig.mask_qc_beam_velocity] = np.nan

    #determine where a 3 beam calculation is possible/needed and do it for the first 4 beams (doesn't care if there is a 5th beam or not)
    threeBeamSolMask = three_beam_mask_calc(vRef)
    vl1 = three_beam_calc(vRef, threeBeamSolMask, convert)
        
    #store results in dataset
    sig['velBeam_l1'] = (('time','z','beam'), vl1)
    sig.velBeam_l1.attrs = sig.beam_velocity_l0.attrs.copy()
    sig.velBeam_l1.attrs['notes'] = 'Velocity in beam coordinates after QC. Uses mask_qc_beam_velocity to set bad data to NaN. also Includes 3-beam solution where needed as noted in mask_3beam_beam_velocity.'

    sig['mask_3beam_beam_velocity'] = (('time','z','beam'),threeBeamSolMask)
    sig.mask_3beam_beam_velocity.attrs['notes'] = 'True where a beam was filled in the 3-beam solution in l1 and greater velocities.'

    if 'BT' in sig.modes:

        #conversion matrix
        convertBT = sig.beam_to_XYZ_conversion_BT.reshape(4,4)

        #pull raw data and apply QC mask to set bad velocities to NaN
        vRef_BT = sig.beam_velocity_l0_BT.data.copy()
        vRef_BT[sig.mask_qc_beam_velocity_BT] = np.nan

        #identify where 3 beam solution is needed and possible
        threeBeamSolMask_BT = three_beam_mask_calc(vRef_BT)

        #consider moving this to the QC mask earlier in processing
        #nan out bottom track velocities that might be reflecting off the surface of the water
        vRef_BT[sig.bottom_track_depth.data<5] = np.nan

        #check again for isolated velocities that we should remove
        indIsolated = np.isnan(vRef_BT[2:,:]) + np.isnan(vRef_BT[:-2,:])
        indIsolated = np.concatenate((np.isnan(vRef_BT[1,:])[None,:],indIsolated, np.isnan(vRef_BT[-1,:])[None,:]),axis=0)
        vRef_BT[indIsolated] = np.nan
        
        #again identify where 3 beam solution is needed and possible
        threeBeamSolMask_BT_2 = three_beam_mask_calc(vRef_BT)

        #using second identification (which is based on all bad data points) calculate 3 beam solutions
        vl1_BT = three_beam_calc(interp_datetime64(sig.time.data,sig.time_BT.data,vRef_BT[...,:4].data,axis=0), threeBeamSolMask_BT_2, convertBT)

        #construct mask to indicate where 3 beam solution was used and why it was used
        mask_BT3beam_velBeam = np.zeros(vRef_BT.shape)
        mask_BT3beam_velBeam[np.logical_and(threeBeamSolMask_BT_2,threeBeamSolMask_BT)] = 1 # 1 where 3 beam solution is used because of QC 
        mask_BT3beam_velBeam[np.logical_and(threeBeamSolMask_BT_2,~threeBeamSolMask_BT)] = 2 # 2 where 3 beam solution is used because of surface bottom tracking

        if 'I' in sig.modes:
            #if there is a 5th beam, we need to create a 5th beam bottom track. three beam calc returns xyz bt velocities and convert_to_XYZ_4beam calculates the reference frame BT velocity from which we can take the Z component for the beam 5 BT.
            velBT_beam5 = three_beam_calc(interp_datetime64(sig.time_I.data,sig.time_BT.data,vRef_BT[...,:4].data,axis=0), threeBeamSolMask_BT_2, convertBT)
            velBT_beam5 = convert_to_XYZ_4beam(velBT_beam5,convertBT,sig.error_velocity_cutoff)[2]  # only need Z component for beam 5

            #create mask to indicate where 3 beam solution was used.
            mask_BT3beam_velBeam_beam5 = np.zeros(velBT_beam5.shape)
            mask_BT3beam_velBeam_beam5[np.any(np.logical_and(threeBeamSolMask_BT_2,threeBeamSolMask_BT),axis=-1)] = 1
            mask_BT3beam_velBeam_beam5[np.any(np.logical_and(threeBeamSolMask_BT_2,~threeBeamSolMask_BT),axis=-1)] = 2

            #add beam 5 info to bottom track variables
            vl1_BT = np.concatenate((vl1_BT, velBT_beam5[:,None]), axis=-1)
            mask_BT3beam_velBeam[...,-1] = mask_BT3beam_velBeam_beam5

        #save relevant variables
        sig['beam_velocity_l2'] = (('time','z','beam'), sig.velBeam_l1.data - vl1_BT[:,None,:])  # Subtract bottom track velocities from beam velocities
        sig.beam_velocity_l2.attrs = sig.velBeam_l1.attrs.copy()
        sig.beam_velocity_l2.attrs['notes'] = 'Velocity in beam coordinates after bottom track correction. ' \
        'basic QC bottom track data interpolated to relevant time-stamp for each beam and bottom tracks above -5m depth removed before 3 beam solutions were calculated and then BT velocity subtracted. ' \
        'For the 5th beam, uses the Z component of the bottom track 4 beam solution. ' \
        'Uses of the 3 beam solution in BT velocities that are subtracted from l1 velocities are noted in mask_3beam_beam_velocity_BT.'

        sig['mask_3beam_beam_velocity_BT'] = (('time','beam'), mask_BT3beam_velBeam)
        sig.mask_3beam_beam_velocity_BT.attrs['notes'] = '0 where no 3 beam solution was used. 1 where a 3-beam solution was used because of QC flag. 2 where 3-beam solution was used because of surface. Used for the bottom track velocity that was used to calculate the l2 velocities.'

        sig['beam_velocity_l1_BT'] = (('time','beam'),vl1_BT)
        sig.beam_velocity_l1_BT.attrs = sig.beam_velocity_l0_BT.attrs.copy()
        sig.beam_velocity_l1_BT.attrs['notes'] = 'Bottom track velocity in beam coordinates after QC. Uses mask_qc_beam_velocity_BT to set bad data to NaN, then deletes data above -5m depth. also Includes 3-beam solution where needed as noted in mask_BT3beam_velBeam.'

    return sig

def transformSignatureBeamVelocities(sig):

    if 'BT' not in sig.modes:
        raise ValueError("Signature dataset does not contain 'BT' mode data. Code not designed to work without BT data.")
    
    convert = sig.beam_to_XYZ_conversion.reshape(4,4)

    vX,vY,vZ,vError = convert_to_XYZ_4beam(sig.beam_velocity_l2.data,convert,sig.error_velocity_cutoff)
    sig['velocity_XYZ_l2'] = (('time','z','direction'), np.moveaxis(np.array([vX,vY,vZ]), 0, -1))
    sig.velocity_XYZ_l2.attrs = sig.beam_velocity_l2.attrs.copy()
    sig.velocity_XYZ_l2.attrs['notes'] = 'Velocity in XYZ direction of instrument after bottom track correction. ' \
    'same processing state as beam_velocity_l2, solved with 4 beam transformation, setting observations with error > error_velocity_cutoff to NaN. ' \
    'uses of the 3 beam solution are noted in mask_3beam_velocity_XYZ.'

    sig['error_velocity_l2'] = (('time','z'), vError)
    sig.error_velocity_l2.attrs = sig.beam_velocity_l2.attrs.copy()
    sig.error_velocity_l2.attrs['notes'] = 'Error velocity from 4 beam transformation using l2 beam velocities. '

    sig['mask_3beam_velocity_XYZ'] = (('time','z','direction'), np.any(sig.mask_3beam_beam_velocity.data[...,:4], axis=-1)[:,:,None].repeat(3,axis=-1))
    sig.mask_3beam_velocity_XYZ.attrs = sig.mask_3beam_beam_velocity.attrs.copy()
    sig.mask_3beam_velocity_XYZ.attrs['notes'] = 'True where 4 beam transformation used a beam that was filled in using the 3-beam solution.'

    convert_BT = sig.beam_to_XYZ_conversion_BT.reshape(4,4)

    vX_BT,vY_BT,vZ_BT,vError_BT = convert_to_XYZ_4beam(sig.beam_velocity_l1_BT.data,convert_BT,sig.error_velocity_cutoff)
    sig['velocity_XYZ_l1_BT'] = (('time','direction'), np.array([vX_BT, vY_BT, vZ_BT]).transpose())
    sig.velocity_XYZ_l1_BT.attrs = sig.beam_velocity_l1_BT.attrs.copy()
    sig.velocity_XYZ_l1_BT.attrs['notes'] = 'Bottom Track velocity in XYZ direction of instrument. ' \
    'Same processing state as beam_velocity_l1_BT, solved with 4 beam transformation, setting observations with error > error_velocity_cutoff to NaN. ' \
    'uses of the 3 beam solution are noted in mask_3beam_velocity_XYZ_BT.'

    sig['error_velocity_BT'] = (('time',), vError_BT)
    sig.error_velocity_BT.attrs = sig.beam_velocity_l1_BT.attrs.copy()
    sig.error_velocity_BT.attrs['notes'] = 'Error velocity from 4 beam transformation using l1 bottom track beam velocities.'

    sig['mask_3beam_velocity_XYZ_BT'] = (('time','direction'), np.any(sig.mask_3beam_beam_velocity_BT.data[...,:4], axis=-1)[:,None].repeat(3,axis=-1))
    sig.mask_3beam_velocity_XYZ_BT.attrs = sig.mask_3beam_beam_velocity_BT.attrs.copy()
    sig.mask_3beam_velocity_XYZ_BT.attrs['notes'] = 'True where 4 beam transformation used a beam that was filled in using the 3-beam solution for BT velocities'

    sig['velENU_l1_BT'] = (('time','direction'), rotate_signature_XYZ_to_ENU(sig.time_BT.data,sig.time_full.data,sig.velocity_XYZ_l1_BT.data,sig.heading_angle.data,sig.roll_angle.data,sig.pitch_angle.data,orientation='up',degrees=True))
    sig.velENU_l1_BT.attrs = sig.velocity_XYZ_l1_BT.attrs.copy()
    sig.velENU_l1_BT.attrs['notes'] = 'Bottom track velocity in ENU coordinates. Calculated from velocity_XYZ_l1_BT using the heading, roll, and pitch of the instrument. ' \
    'mask_3beam_velocity_XYZ_BT also indicates what velocities depend on a 3 beam solution.'
    
    sig['velENU_l2'] = (('time','z','direction'), rotate_signature_XYZ_to_ENU(sig.time.data,sig.time_full.data,sig.velocity_XYZ_l2.data,sig.heading_angle.data,sig.roll_angle.data,sig.pitch_angle.data,orientation='up',degrees=True))
    sig.velENU_l2.attrs = sig.velocity_XYZ_l2.attrs.copy()
    sig.velENU_l2.attrs['notes'] = 'velocity in ENU coordinates. Calculated from velocity_XYZ_l2 using the heading, roll, and pitch of the instrument. ' \
    'mask_3beam_velocity_XYZ also indicates what velocities depend on a 3 beam solution.'

    return sig

def signatureGetBottom(sig,existing_bottom_detection=None,interactive=False):

    if existing_bottom_detection is None:
        sig['bottom_distance_beam'] = (('time','beam'), sig.beam_bottom_track_distance_BT.data.copy())
        sig.bottom_distance_beam.attrs['units'] = 'm'
        sig.bottom_distance_beam.attrs['notes'] = 'Distance to bottom from ADCP beam, calculated using the bottom detection algorithm with bottomDetectionMask, bottomDetectionEdgeMin, bottomDetectionEdgeMax, and bottomDetectionAmplitudeThreshold as inputs.' \
        ' If bottomDetectionSkipped is True for a beam, this is just the raw bottom track distance for that beam.'

        sig['bottomDetectionMask'] = (('time','z','beam'), np.ones(sig.beam_amplitude.shape,dtype=bool))
        sig.bottomDetectionMask.attrs['notes'] = 'Mask for bottom detection, False indicates values excluded from bottom detection algorithm to prevent erroneous detections.'

        sig['bottomDetectionEdgeMin'] = ('beam', np.full(sig.beam.shape,45))
        sig['bottomDetectionEdgeMin'].attrs['notes'] = 'Minimum amplitude canny edge detection used in bottom detection algorithm.'

        sig['bottomDetectionEdgeMax'] = ('beam', np.full(sig.beam.shape,115))
        sig['bottomDetectionEdgeMax'].attrs['notes'] = 'Maximum amplitude canny edge detection used in bottom detection algorithm.'

        sig['bottomDetectionAmplitudeThreshold'] = ('beam', np.full(sig.beam.shape,50))
        sig['bottomDetectionAmplitudeThreshold'].attrs['notes'] = 'Amplitude threshold used in bottom detection algorithm. detected bottom points must be above this threshold.'

        sig['bottomDetectionSkipped'] = ('beam', np.ones(sig.beam.shape,dtype=bool))
        sig['bottomDetectionSkipped'].attrs['notes'] = 'True where bottom detection was skipped for this beam and default bottom tracking was used instead.'

        sig['bottomDetectionMaxInterpolateGap'] = ('beam', np.full(sig.beam.shape, 11))
        sig['bottomDetectionMaxInterpolateGap'].attrs['notes'] = 'Maximum gap size for interpolation in bottom detection algorithm. If a gap larger than this is found, the bottom detection returns nan for that gap.'

        sig['bottomDetectionEdgeBlurKernelX'] = ('beam', np.full(sig.beam.shape, 9))
        sig['bottomDetectionEdgeBlurKernelX'].attrs['notes'] = 'X size of the kernel used to blur the edges detected by canny before using them to filter amplitude for final bottom detection.'

        sig['bottomDetectionEdgeBlurKernelY'] = ('beam', np.full(sig.beam.shape, 3))
        sig['bottomDetectionEdgeBlurKernelY'].attrs['notes'] = 'Y size of the kernel used to blur the edges detected by canny before using them to filter amplitude for final bottom detection.'

        if interactive:
            print('not currently implementing interactive bottom detection')
            # for i in range(sig.beam.size):
            #     sig = interactive_bottom_detection(sig, i)
    else:
        for var in existing_bottom_detection.data_vars:
            if var not in sig.data_vars:
                sig[var] = existing_bottom_detection[var].copy()
            else:
                sig[var].data = existing_bottom_detection[var].data.copy()
                sig[var].attrs.update(existing_bottom_detection[var].attrs)

    return sig

def initial_clean_signature(sigRaw,sig_config,qc,lat,existing_bottom_detection=None,use_interactive=False):
    """
    Cleans and processes the ADCP data from the Dragon before incorporating it into the Dragon dataset.
    This function assumes that sigRaw is an xarray Dataset produced with loadSignature.
    It extracts and renames variables based on the modes present in the data.
    Also does initial cleaning and data processing.
    """
    sigProc = xr.Dataset()

    # Determine which modes are present in the data
    modes = sigRaw.modes
        
    if 'BT' in modes:
        sigProc.attrs['beam_to_XYZ_conversion_BT'] = sig_config.bt_beam2xyz.values.flatten()
        sigProc.attrs['bottom_track_fom_cutoff'] = qc['bottom_track_fom_cutoff']
    if 'IHR' in modes:
        sigProc.attrs['HR_corr_cutoff'] = qc['HR_corr_cutoff']

    sigProc.attrs['beam_angle'] = sig_config.beamConfiguration1_theta.data
    sigProc.attrs['correlation_cutoff'] = qc['correlation_cutoff']
    sigProc.attrs['amplitude_cutoff'] = qc['amplitude_cutoff']
    sigProc.attrs['roll_cutoff'] = qc['roll_cutoff']
    sigProc.attrs['error_velocity_cutoff'] = qc['err_vel_cutoff']
    sigProc.attrs['pitch_cutoff'] = qc['pitch_cutoff']
    sigProc.attrs['frequency'] = str(sig_config.plan_frequency.data) + ' ' + str(sig_config.plan_frequency.units)
    sigProc.attrs['modes'] = sigRaw.modes
    sigProc.attrs['beam_to_XYZ_conversion'] = sig_config.burst_beam2xyz.values.flatten()
    sigProc.attrs['conversion_format'] = '4x4 array flattened row wise (so, row 1, followed by row 2, ...)'

    sigProc = create_signature_coordinates(sigProc,sigRaw,sig_config)

    # Load data into new dataset
    for mode in modes:
        sigProc = load_signature_mode_specific(sigProc, sigRaw, mode)

    sigProc = signature_transpose(sigProc)
    sigProc = combine_signature_full_time_data(sigProc)

    sigProc = pressure_correction_signature(sigProc,lat)

    if 'I' in modes:
        sigProc,errorI = reformat_signature_beam_5_data(sigProc,sigRaw.I_Error.data)

    if 'BT' in modes:
        sigProc,errorBT = reformat_signature_bottom_track_data(sigProc,sigRaw.BT_Error.data)

    sigProc = clean_signature_beam_velocities(sigProc, sigRaw.Error.data, beam5='I' in modes, error5=errorI if 'I' in modes else None, BT='BT' in modes, errorBT=errorBT if 'BT' in modes else None)

    sigProc = calc_signature_beam_directions_and_distance_to_surface(sigProc)
    sigProc = calc_backscatter(sigProc)

    sigProc = signature_velocity_correction(sigProc)

    sigProc = transformSignatureBeamVelocities(sigProc)
    sigProc = signatureGetBottom(sigProc,existing_bottom_detection=existing_bottom_detection,interactive=use_interactive)

    sigProc.attrs['velocity_naming'] = 'Beam indicates beam coordinates, ' \
    'XYZ indicates instrument coordinates, ' \
    'ENU indicates Earth coordinates. ' \
    'l0 indicates raw velocities, ' \
    'l1 indicates beam velocities after initial QC with 3 beam solution replacing eliminated data where possible (and as noted by 3beam masks), ' \
    'l2 indicates velocities after bottom track corrected, using bottom track data interpolated to correct timestamp after QC and before 3beam solutions (BT 3 beam solutions used in these calculations denoted by BT3beam masks). ' \
    'BT indicates bottom track data.'

    return sigProc


def load_raw_signature(fileIn,config,software='signatureDeployment'):
    # Works from the output files of nortek software that has converted from ad2cp to matlab
    # If software = signatureDeployment, assumes the signature deployment software was used
    #   The output is then a .mat file with 4 structs in it.  
    # If software = 'MIDAS', assumes the MIDAS software was used.
    #   The output is then a .mat file with 2 structs in it, and 2 more structs inside the data struct.
    # This code only loads one file at a time, which will need to be combined later separately

    # Modes is a list that contains what modes were used, which sets how many time-stamps there are. 
    # Can currently contain 'normal', 'I', 'IHR', 'Echo1', 'Echo2', and/or 'BT'.
    
    # The code has only been used for 1 deployment with each software, so probably needs to be checked
    # before being used in a new deployment.

    #config is a boolean for whether or not this function should output the configuration

    #region variables we don't want to save
    label1 = np.array(['Burst','IBurstHR','Echo1Bin1_1000kHz','Echo2Bin1_1000kHz','IBurst','BurstBT'])
    label2 = np.array(['AHRSRotationMatrix','MatlabTimeStamp','TimeStamp','Time'])
    #other = np.array([''])

    irrelevant = np.array([[l1+'_'+l2 for l1 in label1] for l2 in label2]).flatten()
    if software == 'MIDAS':
        irrelevant = np.append(irrelevant, ['Units','Comments'])
    #endregion

    #region load structs from matlab file

    f = loadmat(fileIn)

    if software=='signatureDeployment':
        data = f['Data'][0][0]
        con = f['Config'][0][0]
        descriptions = f['Descriptions'][0][0]
        units = f['Units'][0][0]
        timeString = 'Time'
    elif software == 'MIDAS':
        data = f['Data\x00\x00\x00\x00'][0][0]
        con = f['Config\x00\x00'][0][0]
        descriptions = data['Comments'][0][0]
        units = data['Units'][0][0]
        timeString = 'MatlabTimeStamp'
        conUnits = con['Units'][0][0]
        conDescriptions = con['Comments'][0][0]
    #endregion

    #region set up coordinates for dataset (and save attributes in process)

    #create dataset
    signature = xr.Dataset()

    modes = []

    if np.squeeze(con['burst_enable'])==1:
        modes = modes+['normal',]
        if np.squeeze(con['burst_nBeams'])==5:
            modes = modes + ['I',]
    
    if np.squeeze(con['bursthr_enable'])==1:
        modes = modes + ['IHR',]

    if np.squeeze(con['bt_nBeams']) == 4:
        modes = modes + ['BT',]

    if np.squeeze(con['echo_enable']) == 1:
        modes = modes + ['Echo1','Echo2']

    print('modes detected: ')
    print(modes)

    #create time coordinate
    if 'normal' in modes:
        t = np.squeeze(data['Burst_'+timeString])
        t = np.datetime64('0000-01-01','us') + np.timedelta64(1,'D').astype('timedelta64[us]')*t - np.timedelta64(1,'D')
        signature.coords['Time'] = t.astype('datetime64[ns]')
        signature.Time.attrs['timezone'] = 'utc'

    if 'IHR' in modes:
        t = np.squeeze(data['IBurstHR_'+timeString])
        t = np.datetime64('0000-01-01','us') + np.timedelta64(1,'D').astype('timedelta64[us]')*t - np.timedelta64(1,'D')
        signature.coords['HRTime'] = t.astype('datetime64[ns]')
        signature.Time.attrs['timezone'] = 'utc'

    if 'Echo1' in modes:
        t = np.squeeze(data['Echo1Bin1_1000kHz_'+timeString])
        t = np.datetime64('0000-01-01','us') + np.timedelta64(1,'D').astype('timedelta64[us]')*t - np.timedelta64(1,'D')
        signature.coords['Echo1Time'] = t.astype('datetime64[ns]')
        signature.Time.attrs['timezone'] = 'utc'

    if 'Echo2' in modes:
        t = np.squeeze(data['Echo2Bin1_1000kHz_'+timeString])
        t = np.datetime64('0000-01-01','us') + np.timedelta64(1,'D').astype('timedelta64[us]')*t - np.timedelta64(1,'D')
        signature.coords['Echo2Time'] = t.astype('datetime64[ns]')
        signature.Time.attrs['timezone'] = 'utc'

    if 'I' in modes:
        t = np.squeeze(data['IBurst_'+timeString])
        t = np.datetime64('0000-01-01','us') + np.timedelta64(1,'D').astype('timedelta64[us]')*t - np.timedelta64(1,'D')
        signature.coords['ITime'] = t.astype('datetime64[ns]')
        signature.ITime.attrs['timezone'] = 'utc'

    if 'BT' in modes:
        t = np.squeeze(data['BurstBT_'+timeString])
        t = np.datetime64('0000-01-01','us') + np.timedelta64(1,'D').astype('timedelta64[us]')*t - np.timedelta64(1,'D')
        signature.coords['BTTime'] = t.astype('datetime64[ns]')
        signature.BTTime.attrs['timezone'] = 'utc'

    #load configuration info as attributes into dataset as a dataset
    configuration = xr.Dataset()
    configuration.coords['Beam'] = np.arange(1,5)
    configuration.coords['Beam2'] = np.arange(1,5)

    for n in con.dtype.names:
        if software == 'signatureDeployment':
            if n in descriptions.dtype.names or n in units.dtype.names:
                configuration[n] = np.squeeze(con[n])
                if n in descriptions.dtype.names:
                    getattr(configuration,n).attrs['description'] = np.squeeze(descriptions[n])
                if n in units.dtype.names:
                    getattr(configuration,n).attrs['units'] = np.squeeze(units[n])
            elif n == 'Burst_Beam2xyz':
                configuration[n] = (('Beam','Beam2'),np.squeeze(con[n]))
            else:
                configuration.attrs[n] = np.squeeze(con[n])
        elif software == 'MIDAS':
            if n.split('_')[-1] == 'beam2xyz':
                beam2xyz1d = np.squeeze(con[n])
                beam2xyz2d = np.array([beam2xyz1d[4*i:4*i+4] for i in np.arange(4)])
                configuration[n] = (('Beam','Beam2'),beam2xyz2d)
                getattr(configuration,n).attrs['description'] = np.squeeze(conDescriptions[n])
            elif n in conDescriptions.dtype.names or n in conUnits.dtype.names:
                configuration[n] = np.squeeze(con[n])
                if n in conDescriptions.dtype.names:
                    getattr(configuration,n).attrs['description'] = np.squeeze(conDescriptions[n])
                if n in conUnits.dtype.names:
                    getattr(configuration,n).attrs['units'] = np.squeeze(conUnits[n])
            elif n != 'Units' and n != 'Comments' :
                configuration.attrs[n] = np.squeeze(con[n])

    #use attributes to create vertical bin coordinates

    if software == 'signatureDeployment':
        configStrings = ['Burst','BurstHR','EchoSounder']
        cellStrings = ['_NCells','_CellSize','BlankingDistance']
    elif software == 'MIDAS':
        configStrings = ['burst','bursthr','echo']
        cellStrings = ['_nCells','_cellSize','_blankingDistance']

    if 'normal' in modes:
        signature.coords['Z'] = np.arange(1,configuration[configStrings[0]+cellStrings[0]]+1) \
            * configuration[configStrings[0]+cellStrings[1]].data + configuration[configStrings[0]+cellStrings[2]].data
        signature.Z.attrs['reference'] = 'center of bin'

    signature.coords['Beam'] = np.arange(4) + 1
    signature.coords['Beam2'] = np.arange(4) + 1

    if 'IHR' in modes:
        signature.coords['HRZ'] = np.arange(1,configuration[configStrings[1]+cellStrings[0]]+1)\
            * configuration[configStrings[1]+cellStrings[1]] + configuration[configStrings[1]+cellStrings[2]]
        signature.HRZ.attrs['reference'] = 'center of bin'

    if 'Echo1' in modes or 'Echo2' in modes:
        signature.coords['EchoZ'] = np.arange(1,configuration[configStrings[2]+cellStrings[0]]+1)\
            * configuration[configStrings[2]+cellStrings[1]] + configuration[configStrings[2]+cellStrings[2]]
        signature.EchoZ.attrs['reference'] = 'center of bin'

    switcher1 = {}
    for n in signature.coords:
        switcher1[signature[n].size] = n

    switcher2 = {'Burst':'Time','Echo1Bin1':'Echo1Time',\
        'Echo2Bin1':'Echo2Time','IBurstHR':'HRTime','IBurst':'ITime','BurstBT':'BTTime'}

    #loop through each variable in data struct
    for n in data.dtype.names:
        # ignore variables we don't want
        if not n in irrelevant:
            #clean out extra dimensions
            d = np.squeeze(data[n])
            
            #create list of coordinate names based on dimension lengths
            dims = []
            for i in np.shape(d):
                if i in switcher1:
                    if 'Time' in switcher1[i]:
                        dims = dims+[switcher2[n.split('_')[0]]]
                    else:
                        dims = dims+[switcher1[i]]
            
            beamDims = np.where(np.array(dims)=='Beam')[0]
            if np.size(beamDims)>1:
                j=2
                for i in beamDims[1:]:
                    dims[i] = f'Beam{j}'
                    j+=1

            #load into dataset
            signature[n] = (dims,d)

            s = n.split('_')[-1]

            #check if we have the units (really ugly because the matlab file is formatted terribly)
            if n in units.dtype.names:
                getattr(signature,n).attrs['units'] = np.squeeze(units[n])
            elif 'Profile_'+s in units.dtype.names:
                getattr(signature,n).attrs['units'] = np.squeeze(units['Profile_'+s])
            elif 'Sensor_'+s in units.dtype.names:
                getattr(signature,n).attrs['units'] = np.squeeze(units['Sensor_'+s])
            elif s == 'AmbiguityVel':
                getattr(signature,n).attrs['units'] = np.squeeze(units['Profile_AmbiguityVelocity'])
            elif 'Cor' in s:
                getattr(signature,n).attrs['units'] = np.squeeze(units['Profile_Correlation'])
            elif 'Amp' in s:
                getattr(signature,n).attrs['units'] = np.squeeze(units['Profile_Amplitude'])
            elif 'Vel' in s:
                getattr(signature,n).attrs['units'] = np.squeeze(units['Profile_Velocity'])
            elif 'Freq' in s:
                getattr(signature,n).attrs['units'] = 'kHz'
            elif s == 'Soundspeed':
                getattr(signature,n).attrs['units'] = np.squeeze(units['Sensor_SoundSpeed'])
            
    #endregion

    #region rename variables

    namedict = {}
    for n in signature.variables:
        s = n.split('_')
        if s[0] == 'Burst':
            namedict[n] = s[-1]
        elif s[0] == 'Echo1Bin1':
            namedict[n] = 'Echo1_'+s[-1]
        elif s[0] == 'Echo2Bin1':
            namedict[n] = 'Echo2_'+s[-1]
        elif s[0] == 'IBurstHR':
            namedict[n] = 'HR_'+s[-1]
        elif s[0] == 'IBurst':
            namedict[n] = 'I_'+s[-1]
        elif s[0] == 'BurstBT':
            namedict[n] = 'BT_'+s[-1]

    signature = signature.rename(namedict)

    signature.attrs['modes'] = modes
    configuration.attrs['modes'] = modes

    #endregion

    if config==True:
        return signature, configuration
    else:
        return signature



def process_deployment_adcp(deployment_path,processed_path,qc,lat,overwrite=False):
    deployment_num = '_'.join(deployment_path.name.split('_')[:-1])
    sig_paths = find_sig_folders(deployment_path)
    output_dict = {}
    for sig in sig_paths:
        print(f'found {sig.name}\n')

        #find all mat files for this adcp deployment and determine which ones group together
        # adcp_files = list(sig.rglob('*.mat'))
        adcp_files = defaultdict(list)
        for raw_file in sig.rglob('*.mat'):
            # Skips macOS metadata files (._*) that cause error
            if raw_file.name.startswith('._'):
                continue
            adcp_files[raw_file.name.split('.')[0]].append(raw_file)
        # grouped_files = defaultdict(list)
        # adcp_names = [file.name.split('.')[0] for file in adcp_files]
        sig_dict = {}
        #loop through process file that wants to be created
        for file_name, raw_files in adcp_files.items():
            #prep save file path to see if deployment already processed
            proc_save_file = processed_path / f'dragon_{deployment_num}_{sig.name}.nc'
            if not overwrite and proc_save_file.is_file():
                print(f'skipping {proc_save_file.name}, already created')
                sig_dict[file_name] = xr.load_dataset(proc_save_file)
                continue
        
            proc_adcp_list = []
            #loop through all raw files and do initial cleaning
            for raw_file in sorted(raw_files):
                file_number = raw_file.name.split('.')[-2].split('_')[-1]
                raw_save_folder = raw_file.parents[1] / 'netcdf'
                raw_save_folder.mkdir(parents=True,exist_ok=True)
                raw_save_file = raw_save_folder / f'{file_name}_{sig.name}_{file_number}.nc'
                config_save_file = raw_save_folder / f'{file_name}_{sig.name}_{file_number}_config.nc'
                if not overwrite and raw_save_file.is_file():
                    print(f'skipping {raw_file.name} -> {raw_save_file.name}, already converted')
                    print(f'cleaning {raw_save_file.name}\n')
                    with xr.open_dataset(raw_save_file) as raw_adcp, \
                        xr.open_dataset(config_save_file) as adcp_config:
                        proc_adcp_temp = initial_clean_signature(
                            raw_adcp.load(),
                            adcp_config.load(),
                            qc=qc,
                            lat=lat,
                        )
                else:
                    print(f'converting {raw_file.name}')
                    raw_adcp, adcp_config = load_raw_signature(raw_file,config=True,software='MIDAS')
                    file_save(raw_adcp,raw_save_file,overwrite)
                    file_save(adcp_config,config_save_file,overwrite)
                    print(f'cleaning {raw_save_file.name}\n')
                    proc_adcp_temp = initial_clean_signature(raw_adcp,adcp_config,qc=qc,lat=lat)
                proc_adcp_list.append(proc_adcp_temp)
            if not proc_adcp_list:
                raise ValueError(f'no processed adcp data for {file_name}. Probably an error in the code.')
            elif len(proc_adcp_list)>1:
                print('joining multiple processed files into one dataset')
                #identify all time stamps
                order = np.argsort([temp.time.data[0] for temp in proc_adcp_list])
                proc_adcp_list = [proc_adcp_list[i] for i in order]

                dims_to_concatenate = [
                    dim for dim in proc_adcp_list[0].dims
                    if 'time' in dim
                ]

                #concatenate along first time dimension
                proc_adcp = xr.concat([temp.drop_dims(dims_to_concatenate[1:]) for temp in proc_adcp_list], dim=dims_to_concatenate[0],data_vars=None,compat='equals')

                #concatenate along all other time dimensions
                for i in range(1, len(dims_to_concatenate)):
                    for name, da in xr.concat([temp.drop_dims(dims_to_concatenate[:i]+dims_to_concatenate[i+1:]) for temp in proc_adcp_list], dim=dims_to_concatenate[i],data_vars=None,compat='equals').data_vars.items():
                        proc_adcp[name] = da
            else:
                proc_adcp = proc_adcp_list[0]

            file_save(proc_adcp,proc_save_file,overwrite)
            print('')  # noqa: FURB105
            sig_dict[file_name] = proc_adcp
        output_dict[sig.name] = sig_dict

    return output_dict
