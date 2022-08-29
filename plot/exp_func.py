import exp_util as eu
import os
from exp_util import *
from IPython.display import display


def get_table_accuracy_of_estimators_in_default_setting(delta=3, verbose_level=0, **kwargs):
    sep_mark = False
    eu._seeds = [0]
    if "dataNames" in kwargs:
        eu._dataNames = kwargs["dataNames"].copy()

    trad_algs = ['LBS']
    base_algs = ['CardNet', 'DREAM']
    prfx_algs = ['PAstrid', 'PCardNet', 'PDREAM']

    eu._xvals = [1.0]
    eu._xvals_dict = {
        "LBS": [0.1],
    }

    eu._xlabel = "p_train"
    eu._xlabel_dict = {
        "LBS": "p_test",
    }

    eu._ylabels = ['anal:err', 'anal:q50', 'est:q90', 'anal:q99', 'anal:q100']

    eu._default_map_dict = arg_default_map_dict.copy()
    eu._default_map_dict['PAstrid']['delta'] = -delta
    for alg in ['CardNet', 'PCardNet', 'DREAM', 'PDREAM', 'LBS']:
        eu._default_map_dict[alg]['max_d'] = delta

    formatter_dict = default_dict({
        # ('WIKI', 'anal:q100'): lambda x: '-' if np.isnan(x) else int(x),
        # ('IMDB', 'anal:q100'): lambda x: '-' if np.isnan(x) else int(x),
        # ('DBLP', 'anal:q100'): lambda x: '-' if np.isnan(x) else int(x),
        # ('GENE', 'anal:q100'): lambda x: '-' if np.isnan(x) else int(x),
        ('WIKI', 'anal:q10'): lambda x: '-' if np.isnan(x) else '{:.3f}'.format(x),
        ('IMDB', 'anal:q10'): lambda x: '-' if np.isnan(x) else '{:.3f}'.format(x),
        ('DBLP', 'anal:q10'): lambda x: '-' if np.isnan(x) else '{:.3f}'.format(x),
        ('GENE', 'anal:q10'): lambda x: '-' if np.isnan(x) else '{:.3f}'.format(x),
        ('WIKI', 'anal:q100'): lambda x: '-' if np.isnan(x) else '{:.1f}'.format(x),
        ('IMDB', 'anal:q100'): lambda x: '-' if np.isnan(x) else '{:.1f}'.format(x),
        ('DBLP', 'anal:q100'): lambda x: '-' if np.isnan(x) else '{:.1f}'.format(x),
        ('GENE', 'anal:q100'): lambda x: '-' if np.isnan(x) else '{:.1f}'.format(x),
        ('WIKI', 'anal:q99'): lambda x: '-' if np.isnan(x) else '{:.1f}'.format(x),
        ('IMDB', 'anal:q99'): lambda x: '-' if np.isnan(x) else '{:.1f}'.format(x),
        ('DBLP', 'anal:q99'): lambda x: '-' if np.isnan(x) else '{:.1f}'.format(x),
        # ('GENE', 'anal:q99'): '{:.1f}'.format,
    })
    formatter_dict.set_default(lambda x: '-' if np.isnan(x) else '{:.3f}'.format(x))

    eu._algs = trad_algs.copy()
    if len(trad_algs) > 0:
        df_trad = get_df_model_lbs_egr(0)
        df_trad = refine_default_err_df(df_trad, tag='Traditional', formatter_dict=formatter_dict)
        if verbose_level >= 1 and eu._verbose_tbl:
            display(df_trad)

    eu._algs = base_algs.copy()
    df_base = get_df_model(verbose_level=verbose_level)
    df_base = refine_default_err_df(df_base, tag='Base', formatter_dict=formatter_dict)
    if verbose_level >= 1 and eu._verbose_tbl:
        display(df_base)

    eu._algs = prfx_algs.copy()

    df_prfx = get_df_model(verbose_level=verbose_level)
    if verbose_level >= 1 and eu._verbose_tbl:
        display(df_prfx)
    df_prfx = refine_default_err_df(df_prfx, tag='Prefix', formatter_dict=formatter_dict)

    df_list = []
    if not sep_mark:
        if len(df_trad):
            df_list.append(df_trad)
    if len(df_base):
        df_list.append(df_base)
    if len(df_prfx):
        df_list.append(df_prfx)

    df_all = pd.concat(df_list)
    formated_df = rename_df_col_index(df_all).applymap("{:.2f}".format)
    display(formated_df)
    table_key = "_".join(eu._dataNames)
    os.makedirs("htmls/", exist_ok=True)
    formated_df.to_html(f"htmls/estimator_{table_key}.html")


def get_df_model_lbs_egr(verbose_level):
    dataNames = eu._dataNames.copy()
    _default_map = eu._default_map_dict['LBS'].copy()
    if 'GENE' in dataNames:
        eu._dataNames.remove('GENE')
        df1 = get_df_model(verbose_level=verbose_level)
        eu._dataNames = ['GENE']
        eu._default_map_dict['LBS']['PT'] = 1
        eu._default_map_dict['LBS']['Ntbl'] = 8
        df2 = get_df_model(verbose_level=verbose_level)

        df = pd.concat([df1, df2])
        eu._dataNames = dataNames.copy()
        eu._default_map_dict['LBS'] = _default_map.copy()
    else:
        df = get_df_model(verbose_level=verbose_level)
    return df


def exp1_qs_vs_gt(prefix, timeout, delta=None, **kwargs):
    # exp1: join varying sampling ratio
    eu._seeds = [0]

    eu._algs = ['NaiveGen', 'Qgram', 'TASTE', 'TEDDY', 'SODDY']

    eu._dataNames = ['WIKI', 'IMDB', 'DBLP', 'GENE']
    if 'dataNames' in kwargs:
        eu._dataNames = kwargs['dataNames']
    eu._xvals = [0.01, 0.03, 0.1, 0.3, 1.0]
    eu._xlabel = 'pTrain'
    eu._ylabel = 'time'
    eu._ylabels = ["join:time", "join:n_rec", "join:n_qry", "join:n_prfx", "status:status"]
    if prefix:
        prfx = 1
    else:
        prfx = 0

    eu._default_map = {'ps': 1.0, 'pq': 1.0, 'thrs': delta, 'prfx': prfx, 'max_l': 20}

    ofname_pat = f"qry_size_vs_time_{{dataName}}"
    ofname_pat += f"_{delta}"
    if prefix:
        ofname_pat += "_prfx"
    ofname_pat += ".png"

    df = get_df(1)
    df['n_qry'] = df['join:n_prfx'] * (delta + 1)
    df['n_qs'] = df['join:n_prfx']
    df = clipping_df(df, "join:time", max_val=timeout)
    df = aggregate_df(df, agg=['mean', 'std'], counting=True)
    df_list = slice_df_by_dataName(df)

    eu._xrange = (min(eu._xvals)-0.001, max(eu._xvals)+0.01)
    eu._xlabel = 'xval'
    eu._ylabel = 'join:time'
    eu._zlabel = None

    eu._xlog = True
    eu._ylog = True
    eu._xticks = eu._xvals
    eu._yticks = None
    eu._xticks_format = lambda x, pos: percent_foramt_from_ratio(x)

    if prefix:
        plt_args = {
            "WIKI": {
                'yticks': [1e2, 1e3, 1e4, 1e5],
                'ylim': [1e2, 1e5],
                'xlim': [0.03, 1],
                'xminor': False,
                'yminor': False,
            },
            "IMDB": {
                'yticks': [1e3, 1e4],
                'yticks': [1e2, 1e3, 1e4, 1e5],
                'xlim': [0.03, 1],
                'xminor': False,
                'yminor': True,
            },
            "DBLP": {
                'yticks': [1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
                'xlim': [0.03, 1],
                'xminor': False,
                'yminor': True,
            },
            "GENE": {
                'xlim': [0.01, 1],
                'yticks': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
                'xminor': False,
            }
        }
    else:
        plt_args = {
            "WIKI": {
                'yticks': [1e2, 1e3, 1e4, 1e5],
                'xlim': [0.01, 1],
                'ylim': [1e2, 1e5],
                'xminor': False,
                'yminor': False,
            },
            "IMDB": {
                'yticks': [1e2, 1e3, 1e4, 1e5],
                'xlim': [0.01, 1],
                'ylim': [3e1, 1e5],
                'xminor': False,
            },
            "DBLP": {
                'yticks': [1e0, 1e1, 1e2, 1e3, 1e4],
                'xlim': [0.01, 1],
                'xminor': False,
            },
            "GENE": {
                'xlim': [0.01, 1],
                'yticks': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
                'xminor': False,
            }
        }
    plot_df_list(df_list, ofname_pat=ofname_pat, **plt_args)


def sort_multicolumn_df(df, algs, dnames, selected):
    tag_mode = isinstance(df.index[0], tuple)
    if tag_mode:
        tag = df.index[0][0]
    if tag_mode:
        df = df.reindex([(tag, alg) for alg in algs], axis=0)  # index order
    else:
        df = df.reindex(algs, axis=0)  # index order
    df = df.reindex(list(zip(np.array(dnames).repeat(
        len(selected)), selected * len(dnames))), axis=1)  # column order
    return df


def transform_raw_df_with_multicolumns(df, selected=['est:err', 'est:q90']):
    tag_mode = 'tag' in df.columns

    df = df.sort_values(['alg', 'data'])
    if tag_mode:
        df = df.set_index(['tag', 'alg', 'data'])  # .xs(['est:err', 'est:90'], axis=1)
    else:
        df = df.set_index(['alg', 'data'])  # .xs(['est:err', 'est:90'], axis=1)
    df = df.unstack(level=-1)
    df = df[selected]
    df = df.swaplevel(0, 1, axis=1)
    df = df.stack(level=-1)
    df = df.unstack(level=-1)
    df.columns.names = [None, None]  # remove column name
    if tag_mode:
        df.index.names = [None, None]  # remove index name
    else:
        df.index.name = None  # remove index name
    return df


def refine_default_err_df(df, tag=None, formatter_dict=None):
    algs = eu._algs
    dataNames = eu._dataNames
    selected = eu._ylabels  # ['est:err', 'est:q90']

    df = aggregate_df(df)
    if tag is not None:
        df = df.assign(tag=tag)
    df = df.drop(columns=['xlabel', 'xval'])
    if len(df) == 0:
        return df
    df = transform_raw_df_with_multicolumns(df, selected=selected)
    df = sort_multicolumn_df(df, algs, dataNames, selected)

    if len(selected) == 1:
        df.columns = df.columns.droplevel(1)
    return df


def rename_df_col_index(df):
    df = df.rename(index=alg_label_dict)
    df = df.rename(columns=eu.data_label_dict, level=0)
    df = df.rename(columns=eu.table_label_dict, level=1)
    return df
