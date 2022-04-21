import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from natsort import natsorted
import plotly.graph_objects as go

# from Conor Messer: https://github.com/ConorMesser/cnv-methods/blob/main/plot_cnv_profile.py

def plot_acr_static(seg_df, ax, csize,
             segment_colors='difference', sigmas=True, min_seg_lw=2, y_upper_lim=2):
    seg_df, chr_order, chrom_start, col_names = prepare_df(seg_df, csize, suffix='.bp')
    add_background(ax, chr_order, csize, height=7)

    # determine segment colors based on input
    if segment_colors == 'black':
        seg_df['color_bottom'] = '#000000'
        seg_df['color_top'] = '#000000'
    elif segment_colors == 'difference':
        seg_df['color_bottom'], seg_df['color_top'] = calc_color(seg_df, col_names['mu_major'], col_names['mu_minor'])
    elif segment_colors == 'cluster':
        phylogic_color_dict = get_phylogic_color_scale()
        seg_df['color_bottom'] = seg_df['cluster_assignment'].map(phylogic_color_dict)
        seg_df['color_top'] = seg_df['color_bottom']
    else:
        seg_df['color_bottom'] = '#2C38A8'  # blue
        seg_df['color_top'] = '#E6393F'  # red

    # draw segments as lines with default line width
    ax.hlines(seg_df[col_names['mu_minor']].values, seg_df['genome_start'], seg_df['genome_end'],
              color=seg_df['color_bottom'], lw=min_seg_lw)
    ax.hlines(seg_df[col_names['mu_major']].values, seg_df['genome_start'], seg_df['genome_end'],
              color=seg_df['color_top'], lw=min_seg_lw)

    # if sigmas are desired, draw over segments
    if sigmas:
        for _, x in seg_df.iterrows():
            ax.add_patch(patches.Rectangle(
                (x['genome_start'], x[col_names['mu_major']] - x[col_names['sigma_major']]),
                x['genome_end'] - x['genome_start'], 2 * x[col_names['sigma_major']],
                color=x['color_top'],
                alpha=1,
                linewidth=0
            ))
            ax.add_patch(patches.Rectangle(
                (x['genome_start'], x[col_names['mu_minor']] - x[col_names['sigma_minor']]),
                x['genome_end'] - x['genome_start'], 2 * x[col_names['sigma_minor']],
                color=x['color_bottom'],
                alpha=1,
                linewidth=0
            ))

    # layout (can be overridden)
    ax.set_xticks(np.asarray(list(chrom_start.values())[:-1]) + np.asarray(list(csize.values())) / 2)
    ax.tick_params(axis='x', bottom=False)
    ax.set_xticklabels(chr_order, fontsize=10)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(12)
    ax.set_xlim(0, chrom_start['Z'])

    ax.set_yticks(list(range(y_upper_lim + 1)))
    ax.set_yticklabels([str(i) for i in range(y_upper_lim + 1)], fontsize=12)
    ax.set_ylim(-0.05, y_upper_lim + 0.05)
    plt.setp(ax.spines.values(), visible=False)
    ax.spines['left'].set(lw=1, position=('outward', 10), bounds=(0, y_upper_lim), visible=True)
    plt.xlabel("Chromosome")
    plt.ylabel("Allelic Copy Number")


def plot_acr_interactive(seg_df, fig, csize,
                         segment_colors='difference', sigmas=True, min_seg_lw=0.015, y_upper_lim=2, row=0, col=0):
    # fig should have background set to white
    seg_df, chr_order, chrom_start, col_names = prepare_df(seg_df, csize, suffix='.bp')
    add_background(fig, chr_order, csize, row=row+1, col=col+1)

    seg_df['color_bottom_diff'], seg_df['color_top_diff'] = calc_color(seg_df, col_names['mu_major'], col_names['mu_minor'])
    if 'cluster_assignment' in seg_df.columns:
        phylogic_color_dict = get_phylogic_color_scale()
        seg_df['color_bottom_cluster'] = seg_df['cluster_assignment'].map(phylogic_color_dict)
        seg_df['color_top_cluster'] = seg_df['color_bottom_cluster']

    if segment_colors == 'difference':
        seg_df['color_bottom'], seg_df['color_top'] = seg_df['color_bottom_diff'], seg_df['color_top_diff']
    elif segment_colors == 'cluster' and 'cluster_assignment' in seg_df.columns:
        seg_df['color_bottom'], seg_df['color_top'] = seg_df['color_bottom_cluster'], seg_df['color_top_cluster']
    else:
        seg_df['color_bottom'] = '#2C38A8'  # blue
        seg_df['color_top'] = '#E6393F'  # red

    trace_num = len(fig.data)
    seg_df.apply(lambda x: make_cnv_scatter(x, fig, col_names, lw=min_seg_lw, row_num=row+1, sigmas=sigmas), axis=1)

    # modify layout
    fig.update_xaxes(showgrid=False,
                     zeroline=False,
                     tickvals=np.asarray(list(chrom_start.values())[:-1]) + np.asarray(list(csize.values())) / 2,
                     ticktext=chr_order,
                     tickfont_size=10,
                     tickangle=0,
                     range=[0, chrom_start['Z']], 
                     row=row+1, col=col+1)
    fig.update_xaxes(title_text="Chromosome", row=-1, col=1)
    fig.update_yaxes(showgrid=False,
                     zeroline=False,
                     tickvals=list(range(y_upper_lim + 1)),
                     ticktext=[str(i) for i in range(y_upper_lim + 1)],
                     tickfont_size=12,
                     ticks="outside",
                     range=[-0.05, y_upper_lim + 0.05],
                     title_text="Allelic Copy Number", 
                     row=row+1, col=col+1)

    ################
    fig.update_layout(plot_bgcolor='white')  # title=mut_df.iloc[0]['Patient_ID'],

    return trace_num, len(fig.data)


def make_cnv_scatter(series, fig, col_names, lw=0.015, row_num=1, sigmas=False):
    start = series['genome_start']
    end = series['genome_end']
    mu_maj = series[col_names['mu_major']]
    mu_min = series[col_names['mu_minor']]
    sigma = series[col_names['sigma_major']]
    color_maj = series['color_top']
    color_min = series['color_bottom']
    if 'cluster_assignment' in series.index:
        cluster = series['cluster_assignment']
    else:
        cluster = 'NA'

    fig.add_trace(go.Scatter(x=[start, start, end, end],
                  y=[mu_min + sigma, mu_min - sigma, mu_min - sigma, mu_min + sigma],
                  fill='toself', fillcolor=color_min, mode='none',
                  hoverinfo='none', name='cnv_sigma',
                  showlegend=False, visible=sigmas), row=row_num, col=1)
    fig.add_trace(go.Scatter(x=[start, start, end, end],
                  y=[mu_maj + sigma, mu_maj - sigma, mu_maj - sigma, mu_maj + sigma],
                  fill='toself', fillcolor=color_maj, mode='none',
                  hoverinfo='none', name='cnv_sigma',
                  showlegend=False, visible=sigmas), row=row_num, col=1)
    fig.add_trace(go.Scatter(x=[start, start, end, end],
                  y=[mu_min + lw, mu_min - lw, mu_min - lw, mu_min + lw],
                  fill='toself', fillcolor=color_min, mode='none',
                  hoveron='fills', name='cnv',
                  text=f'chr{series["Chromosome"]}:{series["Start.bp"]}-{series["End.bp"]}; '
                       f'CN Minor: {mu_min:.2f} +-{sigma:.4f}; '  # todo make original, so no updating needed
                       f'Cluster: {cluster}; '
                       f'Length: {series["length"]:.2e} ({series["n_probes"]} probes, {series["n_hets"]} het sites)',
                  showlegend=False), row=row_num, col=1)
    fig.add_trace(go.Scatter(x=[start, start, end, end],
                  y=[mu_maj + lw, mu_maj - lw, mu_maj - lw, mu_maj + lw],
                  fill='toself', fillcolor=color_maj, mode='none',
                  hoveron='fills', name='cnv',
                  text=f'chr{series["Chromosome"]}:{series["Start.bp"]}-{series["End.bp"]}; '
                       f'CN Major: {mu_maj:.2f} +-{sigma:.4f}; '  # todo make original so no updating needed
                       f'Cluster: {cluster}; '
                       f'Length: {series["length"]:.2e} ({series["n_probes"]} probes, {series["n_hets"]} het sites)',
                  showlegend=False), row=row_num, col=1)


def update_cnv_scatter_cn(fig, major, minor, sigma, start_trace, end_trace, lw=0.015):
    """Updates y values for CNV traces from start_trace to end_trace given by major/minor lists.
    Critical that original traces were added in order: minor (sigma), major (sigma), minor, major"""
    assert end_trace - start_trace == len(major) * 4 and len(major) == len(minor) == len(sigma)

    for i, (minor_val, major_val, sigma) in enumerate(zip(minor, major, sigma)):
        fig.data[start_trace + 4 * i]['y'] = [minor_val + sigma, minor_val - sigma, minor_val - sigma, minor_val + sigma]
        fig.data[start_trace + 4 * i + 1]['y'] = [major_val + sigma, major_val - sigma, major_val - sigma, major_val + sigma]
        fig.data[start_trace + 4 * i + 2]['y'] = [minor_val + lw, minor_val - lw, minor_val - lw, minor_val + lw]
        fig.data[start_trace + 4 * i + 3]['y'] = [major_val + lw, major_val - lw, major_val - lw, major_val + lw]


def update_cnv_scatter_color(fig, color_minor, color_major, start_trace, end_trace):
    assert end_trace - start_trace == len(color_minor) * 4 == len(color_major) * 4

    for i, (minor_val, major_val) in enumerate(zip(color_minor, color_major)):
        fig.data[start_trace + 4 * i]['fillcolor'] = minor_val
        fig.data[start_trace + 4 * i + 1]['fillcolor'] = major_val
        fig.data[start_trace + 4 * i + 2]['fillcolor'] = minor_val
        fig.data[start_trace + 4 * i + 3]['fillcolor'] = major_val


def update_cnv_scatter_sigma_toggle(fig, sigmas):
    fig.update_traces(dict(visible=sigmas), selector={'name': 'cnv_sigma'})


def add_background(ax, chr_order, csize, height=10**7, row=1, col=1):
    base_start = 0
    chrom_ticks = []
    patch_color = 'white'
    for chrom in chr_order:
        if type(ax) == go.Figure:
            ax.add_vrect(base_start, base_start + csize[chrom], fillcolor=patch_color,
                          opacity=0.1, layer='below', line_width=0, exclude_empty_subplots=False, row=row, col=col)
        else:
            p = patches.Rectangle((base_start, -0.2), csize[chrom], height, fill=True, facecolor=patch_color,
                                  edgecolor=None, alpha=.1)  # Background
            ax.add_patch(p)
        patch_color = 'gray' if patch_color == 'white' else 'white'
        chrom_ticks.append(base_start + csize[chrom] / 2)
        base_start += csize[chrom]


def calc_color(seg_df, major, minor):
    from matplotlib import colors
    cmap = colors.LinearSegmentedColormap.from_list("", ["blue", "purple", "red"])
    color_bottom = seg_df.apply(lambda x: colors.rgb2hex(cmap(
        int(np.floor(max(0, (0.5 - 0.5 * scale_diff(x[major] - x[minor])) * 255))))),
                                          axis=1)
    color_top = seg_df.apply(lambda x: colors.rgb2hex(cmap(
        int(np.floor(min(255, (0.5 + 0.5 * scale_diff(x[major] - x[minor])) * 255))))),
                                       axis=1)
    return color_bottom, color_top


def scale_diff(mu_diff):
    return (7*mu_diff**2) / (7*mu_diff**2 + 10)


def prepare_df(df, csize, suffix='.bp'):
    # discover columns
    if 'mu.major' in df.columns:
        col_names = dict(
            mu_major = 'mu.major',
            mu_minor = 'mu.minor',
            sigma_major = 'sigma.major',
            sigma_minor = 'sigma.minor'
        )
    elif 'hscr.a2' in df.columns:
        col_names = dict(
            mu_major = 'hscr.a2',
            mu_minor = 'hscr.a1',
            sigma_major = 'seg_sigma',  # = tau sigma (not allelic sigma), generally slightly lower
            sigma_minor = 'seg_sigma',  # = tau sigma
        )
    else:
        col_names = None

    chr_order = natsorted(list(csize.keys()))
    chrom_start = {chrom: start for (chrom, start) in
                   zip(np.append(chr_order, 'Z'), np.cumsum([0] + [csize[a] for a in chr_order]))}

    df['Chromosome'] = df['Chromosome'].astype(str)
    df = df[df['Chromosome'].isin(chr_order)]

    df[f'Start{suffix}'] = df[f'Start{suffix}'].astype(int)
    df[f'End{suffix}'] = df[f'End{suffix}'].astype(int)
    df['genome_start'] = df.apply(lambda x: chrom_start[str(x['Chromosome'])] + x[f'Start{suffix}'], axis=1)
    df['genome_end'] = df.apply(lambda x: chrom_start[str(x['Chromosome'])] + x[f'End{suffix}'], axis=1)

    return df, chr_order, chrom_start, col_names


def get_hex_string(c):
    return '#{:02X}{:02X}{:02X}'.format(*c)