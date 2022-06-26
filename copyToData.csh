#!/bin/csh

rsync -uv figures/* ssh.swc.ucl.ac.uk:dev/work/ucl/gatsby-swc/fwg/lds_repo/figures
rsync -uv results/* ssh.swc.ucl.ac.uk:dev/work/ucl/gatsby-swc/fwg/lds_repo/results
rsync -uv metadata/* ssh.swc.ucl.ac.uk:dev/work/ucl/gatsby-swc/fwg/lds_repo/metadata
