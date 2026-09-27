
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

METHODS = ['Local-only', 'Non-Collaborative Offloading', 'Random Decision',
           'Greedy', 'Exhaustive Search', 'Random Offloading', 'Proposed MCOR-TD3']
COLORS = ['#e65116', '#007ac3', '#7dab2e', '#ffae00', '#91249f', '#34bceb', '#af1239']
LINES = ['--', '--', '--', '--', ':', ':', '-']
MARKERS = ['o', '^', 's', 'd', 'v', '*', 'o']
X = [2, 5, 8, 12, 16, 20]
DELAY_DATA = {
    'mec_delay': ('Number of MEC Servers', [
        [42,33,32,27,23,19], [44,34,32,28,25,24], [40,35,30,25,22,26],
        [38,30,26,22,20,22], [35,32,27.5,24,22,20], [30,22,20,19,17,16],
        [25,24,22,20,18,15]]),
    'mec_computing_capacity': ('MEC Server Computing Capacity (GHz)', [
        [46,42,44,36,33,29], [44,40,38,41,32,28], [43,39,41,34,31,27],
        [41,36.5,35,38,30,25.5], [39,36,37,32,28.5,24.5], [37,34,33,31,27,23],
        [35,32,30,28,24,20]]),
    'device_delay': ('Number of Devices', [
        [42,30,35,27,24,18], [40,36,33,28,23,25], [35,34,32,26,25,22],
        [34,33,28,24,22,20], [33,32,30,27,20,18], [25,28,25,22,18,19],
        [30,27,23,21,19,15]])}
ANCHORS = [0,10,20,35,50,65,80,100,120,140,160,180,200,220,240,260,280,300]
TRENDS = [
    [-14,-12.4,-11.5,-10.8,-10.5,-10.3,-8.8,-7.4,-8.5,-8.1,-7,-6.7,-7.8,-7.5,-6.5,-7.2,-7.5,-7],
    [-12,-10.8,-9,-7.4,-6.4,-5.6,-4.5,-5,-4.4,-3.7,-4.4,-4.5,-3.2,-4.3,-4.8,-3.5,-3.9,-4.4],
    [-12.3,-10.6,-8.6,-6.8,-6.5,-5.8,-5.8,-3.6,-3.9,-4.5,-4.6,-4,-3,-3.9,-5.3,-4.6,-3.5,-3.2],
    [-10.6,-8,-5.5,-4.3,-3.4,-3.3,-2.6,-1.6,-1.1,-1.5,-2.3,-1.8,-.8,-1,-2,-2.1,-1.4,-1.4]]


def set_metric_plot_style(font_size=14):
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
        'font.size':font_size, 'axes.labelsize':font_size, 'axes.titlesize':font_size,
        'xtick.labelsize':font_size, 'ytick.labelsize':font_size, 'legend.fontsize':font_size,
        'font.weight':'normal', 'axes.labelweight':'normal', 'axes.titleweight':'normal',
        'figure.facecolor':'white', 'axes.facecolor':'white', 'svg.fonttype':'none'})


def new_axes(note, wide=False):
    fig, ax = plt.subplots(figsize=(11 if wide else 9, 6))
    fig.subplots_adjust(left=.13, right=.97, bottom=.20, top=.95)
    fig.text(.5, .025, note, ha='center', va='bottom', fontsize=10, color='#555555')
    ax.tick_params(which='both', direction='in', top=False, right=False,
                   color='#888888', labelcolor='#303030', length=4)
    for side in ('top','right'):
        ax.spines[side].set_visible(False)
    for side in ('left','bottom'):
        ax.spines[side].set_color('#929292')
        ax.spines[side].set_linewidth(.8)
    ax.grid(False)
    return fig, ax


def get_mec_metric_data():
    """Return independent copies of the supplied estimated delay data."""
    return {name: {'xlabel': label, 'x': list(X), 'methods': list(METHODS),
                   'values': [list(row) for row in values], 'provenance':'visual estimates'}
            for name, (label, values) in DELAY_DATA.items()}


def plot_reward_convergence():
    fig, ax = new_axes('Visual approximation only; interpolated trends with synthetic noise.')
    rng, x = np.random.default_rng(18), np.arange(301)
    styles = [('16 TDs','#e65100','-','o',.48), ('32 TDs','#ff0000','--','s',.38),
              ('64 TDs','#ffdf32','--','d',.43), ('128 TDs','#0000ff',':','^',.29)]
    for values, (label, color, ls, marker, noise) in zip(TRENDS, styles):
        y = np.interp(x, ANCHORS, values)+rng.normal(0,noise,len(x))
        y[0], y[-1] = values[0], values[-1]
        ax.plot(x,y,label=label,color=color,linestyle=ls,linewidth=1.25,marker=marker,
                markevery=20,markersize=5.5,markerfacecolor='none',markeredgewidth=1.1)
    ax.set(xlim=(0,300),ylim=(-15,0),xlabel='Training Episodes',ylabel='Reward Value')
    ax.set_xticks(np.arange(0,301,50))
    ax.set_yticks([-15,-10,-5,0])
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.legend(loc='lower right',frameon=False,handlelength=3,labelspacing=.12)
    return fig, ax


def plot_delay_comparison(kind):
    if kind not in DELAY_DATA:
        raise ValueError(f'Unknown delay plot: {kind}')
    xlabel, values = DELAY_DATA[kind]
    fig, ax = new_axes('Visual estimates supplied by the user; not original experimental data.')
    for i, (label, y) in enumerate(zip(METHODS, values)):
        line, = ax.plot(X,y,label=label,color=COLORS[i],linestyle=LINES[i],marker=MARKERS[i],
            linewidth=2.3 if i==6 else 1.7,markersize=9 if i==5 else 6.8,
            markerfacecolor=COLORS[i] if i>=5 else 'none',markeredgewidth=1.4)
        if LINES[i]=='--':
            line.set_dashes([3.5,3.5])
    ax.set(xlim=(0,20),ylim=(0,50),xlabel=xlabel,ylabel='Task Completion Delay')
    ax.set_xticks(np.arange(0,21,2))
    ax.set_yticks(np.arange(0,51,5))
    ax.legend(loc='upper right' if kind=='mec_delay' else 'lower left',
              frameon=False,handlelength=2.6,handletextpad=.3,labelspacing=.12)
    return fig, ax


def read_csv(path, required):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: required columns are {required}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: file has no data rows.')
    return rows


def numbers(rows, key):
    data = np.array([float(row[key]) for row in rows])
    if not np.isfinite(data).all():
        raise ValueError(f'Non-finite values in {key}')
    return data


def plot_training_log(path):
    rows = read_csv(path, ['episode','return'])
    episodes, returns = numbers(rows,'episode'), numbers(rows,'return')
    if len(np.unique(episodes)) != len(episodes) or (episodes < 0).any():
        raise ValueError('Training episode indices must be unique and nonnegative.')
    order = np.argsort(episodes)
    fig, ax = new_axes('Recorded training log; raw episode return, without smoothing or rescaling.')
    ax.plot(episodes[order],returns[order],color='#af1239',linewidth=1.5)
    ax.set(xlabel='Training Episode',ylabel='Episode Return (sum of rewards)')
    return fig, ax


def plot_evaluation_log(path):
    metrics = [('mean_delay_s','Mean Task Completion Delay (s)'),
               ('mean_slot_max_delay_s','Mean Per-slot Maximum Delay (s)'),
               ('cache_hit_ratio_all_requests','Cache Hit Ratio (all requests)')]
    rows = read_csv(path,['policy', *(key for key,_ in metrics)])
    policies = list(dict.fromkeys(row['policy'] for row in rows))
    figures = {}
    for key, label in metrics:
        values = [numbers([r for r in rows if r['policy']==policy], key) for policy in policies]
        if any((v<0).any() or (key.startswith('cache') and (v>1).any()) for v in values):
            raise ValueError(f'Invalid range in {key}')
        fig, ax = new_axes('Recorded evaluation: mean across episodes; error bars = sample SD (not CI).', wide=True)
        x = np.arange(len(policies))
        ax.bar(x,[v.mean() for v in values],color=[COLORS[i%7] for i in x],width=.6)
        # Do not present a zero uncertainty estimate for a single observation.
        for i, v in enumerate(values):
            if len(v)>1:
                ax.errorbar(i,v.mean(),yerr=v.std(ddof=1),fmt='none',ecolor='#333333',capsize=4)
        ax.set_xticks(x, policies, rotation=15, ha='right')
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        figures[key] = (fig, ax)
    return figures


def plot_all_metric_figures(mode='estimated', run_dir=None):
    if mode=='estimated':
        return {'reward_graph':plot_reward_convergence(),
                **{key:plot_delay_comparison(key) for key in DELAY_DATA}}
    if mode!='logs' or run_dir is None:
        raise ValueError('Use estimated or logs with run_dir.')
    root, figures = Path(run_dir), {}
    if (root/'training.csv').exists():
        figures['training_return'] = plot_training_log(root/'training.csv')
    if (root/'evaluation.csv').exists():
        figures.update(plot_evaluation_log(root/'evaluation.csv'))
    if not figures:
        raise FileNotFoundError(f'No training.csv or evaluation.csv found in {root}')
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode',choices=['estimated','logs'],default='estimated')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--output',type=Path,default=Path('mec_figures'))
    parser.add_argument('--dpi',type=int,default=600)
    parser.add_argument('--font-size',type=float,default=14)
    parser.add_argument('--show',action='store_true')
    args = parser.parse_args()
    if args.dpi<=0 or not np.isfinite(args.font_size) or args.font_size<=0:
        parser.error('dpi and font-size must be positive.')
    if args.mode=='logs' and args.run_dir is None:
        parser.error('--run-dir is required for logs mode.')
    set_metric_plot_style(args.font_size)
    figures = plot_all_metric_figures(args.mode,args.run_dir)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,(fig,_) in figures.items():
        fig.savefig(args.output/f'{name}_{args.dpi}dpi.png',dpi=args.dpi,facecolor='white')
        fig.savefig(args.output/f'{name}.svg',facecolor='white')
    (args.output/'PLOT_PROVENANCE.txt').write_text(__doc__,encoding='utf-8')
    print(f'Saved {len(figures)} figures as PNG and SVG to {args.output.resolve()}')
    if args.show:
        plt.show()
    else:
        for fig,_ in figures.values():
            plt.close(fig)


if __name__=='__main__':
    main()
