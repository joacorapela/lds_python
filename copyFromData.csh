#!/bin/csh

rsync -uv "ssh.swc.ucl.ac.uk:dev/work/ucl/gatsby-swc/fwg/lds_repo/figures/*" figures/
rsync -uv "ssh.swc.ucl.ac.uk:dev/work/ucl/gatsby-swc/fwg/lds_repo/results/*" results/ 
rsync -uv "ssh.swc.ucl.ac.uk:dev/work/ucl/gatsby-swc/fwg/lds_repo/metadata/*" metadata/ 
