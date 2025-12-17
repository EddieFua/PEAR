from pyliftover import LiftOver
import pandas as pd
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
parser.add_argument("--chrom_col", default="chromosome")
parser.add_argument("--pos_col", default="base_pair_location")
args = parser.parse_args()

sumstats = pd.read_csv(args.path, sep='\t', compression='gzip', low_memory=False)
lo = LiftOver('hg38', 'hg19')


def liftover_pos(chr_, pos):
    if pd.isna(chr_) or pd.isna(pos):
        return None
    c = str(chr_).strip()
    if c.lower().startswith('chr'):
        c = c[3:]
    if c in ['X', 'Y', 'MT', 'M']:
        chrom = 'chr' + c.replace('MT', 'M')
    else:
        try:
            chrom = 'chr' + str(int(c))  # 1,2,...,22
        except ValueError:
            return None

    try:
        pos0 = int(pos) - 1
        res = lo.convert_coordinate(chrom, pos0)
        if res:
            new_pos0 = res[0][1]      # 0-based
            return int(new_pos0) + 1  # 转回 1-based
        else:
            return None
    except Exception:
        return None

sumstats['pos37'] = sumstats.apply(
    lambda row: liftover_pos(row[args.chrom_col], row[args.pos_col]),
    axis=1
)
sumstats = sumstats.dropna(subset=['pos37'])
sumstats['pos'] = sumstats['pos37'].astype(int)
out_path = f"{os.path.splitext(os.path.splitext(args.path)[0])[0]}_GRCh37.txt.gz"
sumstats.to_csv(out_path, sep='\t', index=False, compression='gzip')