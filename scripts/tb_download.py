import argparse
import shutil
import gzip
import io
import os
import tqdm
from ftplib import FTP


def _download_file(ftp, remote_fname, dest):
    # Use buffer to store gziped data
    buf = io.BytesIO()
    ftp.retrbinary(f"RETR {remote_fname}", buf.write)
    buf.seek(0)
    # Unzip and save to disk
    with gzip.open(buf, "rb") as gfd, open(dest, "wb") as fd:
        shutil.copyfileobj(gfd, fd)
    buf.close()


def _download_year(ftp, ydir, dest_root, overwrite=False):
    dest_dir = os.path.join(dest_root, ydir)
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    ftp_files = [f for f in ftp.nlst() if not f.startswith(".")]
    for f in tqdm.tqdm(ftp_files, ncols=80):
        dest = os.path.join(dest_dir, os.path.splitext(f)[0])
        if not overwrite and os.path.isfile(dest):
            continue
        _download_file(ftp, f, dest)


def tb_ftp_download(host, ftp_dir, dest_root, overwrite=False):
    if not os.path.isdir(dest_root):
        os.makedirs(dest_root)
    print(f"Connecting to host: '{host}'")
    ftp = FTP(host)
    ftp.login()
    ftp.cwd(ftp_dir)
    year_dirs = [d for d in ftp.nlst() if not d.startswith(".")]
    for ydir in year_dirs:
        print(f"Downloading data dir: '{ydir}'")
        ftp.cwd(ydir)
        _download_year(ftp, ydir, dest_root, overwrite)
        ftp.cwd("..")
    ftp.close()


def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-O",
        "--overwrite",
        action="store_true",
        help="Overwrite files if already present",
    )
    p.add_argument("host", help="The ftp host name")
    p.add_argument(
        "remote_dir", help="The root data directory on the remote ftp server"
    )
    p.add_argument(
        "dest_root", help="The destination directory to place data in"
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    tb_ftp_download(args.host, args.remote_dir, args.dest_root, args.overwrite)
