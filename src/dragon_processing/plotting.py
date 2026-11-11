import matplotlib.pyplot as plt
import numpy as np
from cmocean import cm
from matplotlib import gridspec
from scipy.ndimage import convolve1d

from .utilities import dragon_select_cast


def plot_dragon_cast_splitting(data):
    fig, ax = plt.subplots(1,1)
    ax.plot(data.time,data.depth)
    for cast in np.unique(data.cast.data.astype('int')):
        temp_down = dragon_select_cast(data,cast)
        temp_up = dragon_select_cast(data,cast+0.5)
        ax.axvspan(xmin=temp_down.time.data[0], xmax=temp_up.time.data[-1], alpha=0.3,label=f'cast {cast}')
        ax.axvline(x=temp_down.time.data[-1])

    ax.set_xlabel('time')
    ax.set_ylabel('depth')
    ax.yaxis.set_inverted(True)
    
    return fig

def plot_beam_variable(data,variable_name,cmap,title=None,vmin=None,vmax=None,with_bottom_track=False,distance_max = None,depth_min=-3,depth_max=None):

    n_beams = data.adcp_beam.size

    if not depth_max:
        depth_max = np.nanmax(data.beam_depth.data)

    if not distance_max:
        distance_max = np.nanmax(data.beam_distance.data)

    fig = plt.figure(figsize=(9,(9/n_beams)*(depth_max-depth_min)/(distance_max)),dpi=130,layout='constrained')
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0, hspace=0.0)
    gs = gridspec.GridSpec(1,n_beams+1,width_ratios = [1,]*n_beams+[0.1,],wspace=0,figure=fig)

    profile_axes = [fig.add_subplot(gs[0])]
    profile_axes += [fig.add_subplot(gs[i],sharey=profile_axes[0],sharex=profile_axes[0]) for i in range(1,n_beams)]
    cbar_ax = fig.add_subplot(gs[n_beams])

    for i,ax in enumerate(profile_axes):
        pc=ax.tripcolor(
            data.bin.data[np.newaxis,:].repeat(data.time.size,axis=0).flatten(),
            data.beam_depth[:,:,i].data.flatten(),
            data[variable_name].data[:,:,i].flatten(),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap
            )

        if with_bottom_track:
            ax.plot(data.bottom_distanceBeam,data.bottom_track_depth.data[:,i],color='k')

        ax.set_title(f'Beam {i+1}')
        ax.set_xlabel('Distance (m)')
        ax.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(pc,cax=cbar_ax)
    cbar.set_label(f'{variable_name} ({data[variable_name].attrs["units"]})')

    fig.suptitle(title)
    if distance_max:
        profile_axes[0].set_xlim(0,distance_max)
    if depth_max:
        profile_axes[0].set_ylim(depth_max,depth_min)
    else:
        profile_axes[0].yaxis.set_inverted(True)

    profile_axes[0].set_ylabel('Depth (m)')

    return fig

def plot_gridded_variable(data, variable_name, cmap, label_dim, title=None, vmin=None,vmax=None,distance_max=None,depth_min=-3,depth_max=None,vertical_smoothing_window=None):

    n_profiles = data[label_dim].size

    fig = plt.figure(figsize=(9,(9/n_profiles)*(depth_max-depth_min)/(distance_max)),dpi=130,layout='constrained')
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0, hspace=0.0)
    gs = gridspec.GridSpec(1,n_profiles+1,width_ratios = [1,]*n_profiles+[0.1,],wspace=0,figure=fig)

    profile_axes = [fig.add_subplot(gs[0])]
    profile_axes += [fig.add_subplot(gs[i],sharey=profile_axes[0],sharex=profile_axes[0]) for i in range(1,n_profiles)]
    cbar_ax = fig.add_subplot(gs[n_profiles])

    if vertical_smoothing_window:
        data_for_plotting = data[variable_name].rolling(time=vertical_smoothing_window,center=True,min_periods=1).mean(skipna=True).data
    else:
        data_for_plotting = data[variable_name].data

    for i,ax in enumerate(profile_axes):
        pc=ax.pcolormesh(
            data.dist.data,
            data.depth.data,
            data_for_plotting[:,:,i],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap
        )
        ax.set_title(data[label_dim].data[i])
        ax.set_xlabel('Distance (m)')
        ax.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(pc,cax=cbar_ax)
    cbar.set_label(f'{variable_name} ({data[variable_name].attrs["units"]})')

    fig.suptitle(title)
    if distance_max:
        profile_axes[0].set_xlim(0,distance_max)
    if depth_max:
        profile_axes[0].set_ylim(depth_max,depth_min)
    else:
        profile_axes[0].yaxis.set_inverted(True)
    profile_axes[0].set_ylabel('Depth (m)')

    return fig

def plot_field(data,coordinate_system,title=None, vmin_vel=-0.3,vmax_vel=0.3, vmin_scatter=None,vmax_scatter=None, cmap_vel=cm.balance,cmap_scatter = cm.turbid):

    x_range = np.nanmax(data.field_x)-np.nanmin(data.field_x)
    y_range = np.nanmax(data.field_depth)-np.nanmin(data.field_depth)
    fig = plt.figure(figsize=(9,(9/3)*(y_range/x_range)),dpi=130,layout='constrained')
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0, hspace=0.0)
    gs = gridspec.GridSpec(1,6,width_ratios = [1,1,1,1,0.1,0.1],wspace=0,figure=fig)

    profile_axes = [fig.add_subplot(gs[0])]
    profile_axes += [fig.add_subplot(gs[i],sharey=profile_axes[0]) for i in range(1,4)]
    cbar_axes = [fig.add_subplot(gs[i]) for i in range(4,6)]

    if coordinate_system == 'ENU':
        variable_name = 'field_velocity_1_enu'
        labels = ['East','North','Vertical']
    elif coordinate_system == 'XYZ':
        variable_name = 'field_velocity_1_xyz'
        labels = ['Along-Ice', 'Cross-Ice','Vertical']
    else:
        raise ValueError('coordinate_system must be ENU or XYZ')

    for i,ax in enumerate(profile_axes[:-1]):
        pc_vel = ax.pcolormesh(
            data.field_x.data,
            data.field_depth.data,
            data[variable_name].data[:,:,i],
            vmin=vmin_vel,
            vmax=vmax_vel,
            cmap=cmap_vel
        )
        ax.set_title(labels[i])
        ax.set_xlabel('Distance (m)')
        ax.axis('equal')

    pc_scatter = profile_axes[-1].pcolormesh(
        data.field_x.data,
        data.field_depth.data,
        data['field_backscatter'].data,
        vmin=vmin_scatter,
        vmax=vmax_scatter,
        cmap=cmap_scatter
    )

    max_depth = np.nanmax(data.depth)+5

    profile_axes[-1].set_title('Backscatter')
    profile_axes[-1].set_xlabel('Distance (m)')
    profile_axes[-1].axis('equal')

    plt.colorbar(pc_vel,cax=cbar_axes[0],location='right').set_label('Velocity (m/s)',labelpad=2)
    plt.colorbar(pc_scatter,cax=cbar_axes[1],location='right').set_label('Relative Backscatter',labelpad=2)

    cbar_vel = plt.colorbar(pc_vel,cax=cbar_axes[0])
    cbar_vel.set_label('Velocity (m/s)')

    cbar_scatter = plt.colorbar(pc_scatter,cax=cbar_axes[1])
    cbar_scatter.set_label('Backscatter')

    fig.suptitle(title)
    # profile_axes[0].yaxis.set_inverted(True)
    profile_axes[0].set_ylim([max_depth+2,0])
    profile_axes[0].set_ylabel('Depth (m)')

    return fig

def initial_dragon_plots(dragon,save_folder,deploy_name,beam_vel_lim = 0.15,vel_ENU_lim = 0.15,field_vel_lim = 0.15,depth_min=-3,depth_max = None,distance_max=None,vertical_smoothing_window=None):

    field_calculated = 'field_x' in dragon.dims

    beam_vars_to_plot = {
        'beam_correlation' : {
            'vmin' : dragon.adcp_correlation_cutoff,
            'vmax' : np.nanmax(dragon.beam_correlation),
            'cmap' : cm.turbid_r,
            'variable' : 'beam_correlation'
        },
        'beam_amplitude' : {
            'vmin' : dragon.adcp_amplitude_cutoff,
            'vmax' : np.nanmax(dragon.beam_amplitude),
            'cmap' : cm.amp,
            'variable' : 'beam_amplitude'
        },
        'beam_velocity' : {
            'vmin' : -beam_vel_lim,
            'vmax' : beam_vel_lim,
            'cmap' : cm.balance,
            'variable' : 'beam_velocity_l2'
        }
    }

    if field_calculated:
        field_backscatter_min = np.nanmin(dragon.field_backscatter)
        field_backscatter_max = np.nanmax(dragon.field_backscatter)

    save_folder.mkdir(parents=True,exist_ok=True)

    fig = plot_dragon_cast_splitting(dragon)
    fig.savefig(save_folder / 'cast_splitting_overview.png')
    plt.close(fig)

    for cast_num in dragon.cast.data:
        cast = dragon_select_cast(dragon,cast_num)
        for name, var in beam_vars_to_plot.items():
            fig = plot_beam_variable(
                cast,
                var['variable'],
                var['cmap'],
                vmin=var['vmin'],
                vmax=var['vmax'],
                title=f'{deploy_name}_cast_{cast_num}',
                distance_max=distance_max,
                depth_min=depth_min,
                depth_max=depth_max,
            )

            fig.savefig(save_folder / f'cast_{cast_num}_{name}.png')
            plt.close(fig)

        fig = plot_gridded_variable(
            cast,
            'velENU_l2',
            cm.balance,
            label_dim='ENU',
            vmin=-vel_ENU_lim,
            vmax=vel_ENU_lim,
            title=f'{deploy_name}_cast_{cast_num}',
            depth_min=depth_min,
            depth_max=depth_max,
            distance_max=distance_max,
            vertical_smoothing_window=vertical_smoothing_window
        )

        fig.savefig(save_folder / f'cast_{cast_num}_vel_ENU.png')
        plt.close(fig)

        if field_calculated:
            fig = plot_field(
                cast,
                'ENU',
                title=f'{deploy_name}_cast_{cast_num}',
                vmin_vel= -field_vel_lim,
                vmax_vel=field_vel_lim,
                vmin_scatter=field_backscatter_min,
                vmax_scatter=field_backscatter_max,
                cmap_vel = cm.balance,
                cmap_scatter = cm.turbid
            )
            fig.savefig(save_folder / f'cast_{cast_num}_field_ENU.png')
            plt.close(fig)

            fig = plot_field(
                cast,
                'XYZ',
                title=f'{deploy_name}_cast_{cast_num}',
                vmin_vel= -field_vel_lim,
                vmax_vel=field_vel_lim,
                vmin_scatter=field_backscatter_min,
                vmax_scatter=field_backscatter_max,
                cmap_vel = cm.balance,
                cmap_scatter = cm.turbid
            )
            fig.savefig(save_folder / f'cast_{cast_num}_field_XYZ.png')
            plt.close(fig)

