import pandas as pd
from frozendict import frozendict as fdict
import functools


def get_hex_string(c):
    return '#{:02X}{:02X}{:02X}'.format(*c)


def cluster_color(v=None):
    phylogic_color_list = [[166, 17, 129],
                           [39, 140, 24],
                           [103, 200, 243],
                           [248, 139, 16],
                           [16, 49, 41],
                           [93, 119, 254],
                           [152, 22, 26],
                           [104, 236, 172],
                           [249, 142, 135],
                           [55, 18, 48],
                           [83, 82, 22],
                           [247, 36, 36],
                           [0, 79, 114],
                           [243, 65, 132],
                           [60, 185, 179],
                           [185, 177, 243],
                           [139, 34, 67],
                           [178, 41, 186],
                           [58, 146, 231],
                           [130, 159, 21],
                           [161, 91, 243],
                           [131, 61, 17],
                           [248, 75, 81],
                           [32, 75, 32],
                           [45, 109, 116],
                           [255, 169, 199],
                           [55, 179, 113],
                           [34, 42, 3],
                           [56, 121, 166],
                           [172, 60, 15],
                           [115, 76, 204],
                           [21, 61, 73],
                           [67, 21, 74],  # Additional colors, uglier and bad
                           [123, 88, 112],
                           [87, 106, 46],
                           [37, 66, 58],
                           [132, 79, 62],
                           [71, 58, 32],
                           [59, 104, 114],
                           [46, 107, 90],
                           [84, 68, 73],
                           [90, 97, 124],
                           [121, 66, 76],
                           [104, 93, 48],
                           [49, 67, 82],
                           [71, 95, 65],
                           [127, 85, 44],  # even more additional colors, gray
                           [88, 79, 92],
                           [220, 212, 194],
                           [35, 34, 36],
                           [200, 220, 224],
                           [73, 81, 69],
                           [224, 199, 206],
                           [120, 127, 113],
                           [142, 148, 166],
                           [153, 167, 156],
                           [162, 139, 145],
                           [0, 0, 0]]  # black
    colors_dict = {str(i): get_hex_string(c) for i, c in enumerate(phylogic_color_list)}

    if v:
        return colors_dict[str(v)]
    else:
        return colors_dict


def get_unique_identifier(row, chrom='Chromosome', start_pos='Start_position',
                          ref='Reference_Allele', alt='Tumor_Seq_Allele'):
    """Generates unique string for this mutation, including contig, start position, ref and alt alleles.

    Does not include End Position, for this field is not present in mut_ccfs Phylogic output. However, specification of both the alt and ref alleles are enough to distinguish InDels.

    :param row: pd.Series giving the data for one mutation from a maf or maf-like dataframe
    :param chrom: the name of the contig/chromosome column/field; default: Chromosome
    :param start_pos: the name of the start position column/field; default: Start_position
    :param ref: the name of the reference allele column/field; default: Reference_Allele
    :param alt: the name of the alternate allele column/field; default: Tumor_Seq_Allele
    """
    return f"{row[chrom]}:{row[start_pos]}{row[ref]}>{row[alt]}"


@functools.lru_cache(maxsize=32)
def cached_read_csv(fn, **kwargs):
    """Convenience method: Pandas read_csv with a cache already implemented.

    Maxsize is set to 32 by default. At most, 32 function calls will be stored in memory.

    :param fn: filename/path
    :param kwargs: additional arguments to be passed to pandas.read_csv
    :return: pandas.DataFrame from given filename
    """
    return pd.read_csv(fn, **kwargs)


def freezeargs(func):
    """Transform mutable dictionary into immutable and lists into tuples

    Useful to be compatible with cache. Use: decorate functions (@freezeargs) that are decorated with a
    functools cache but have mutable inputs.

    :param func: function that is being wrapped
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([fdict(arg) if isinstance(arg, dict) else (tuple(arg) if isinstance(arg, list) else arg) for arg in args])
        kwargs = {k: fdict(v) if isinstance(v, dict) else (tuple(v) if isinstance(v, list) else v) for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped