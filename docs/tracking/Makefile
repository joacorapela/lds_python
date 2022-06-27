all: tracking.pdf

%.pdf: %.tex
	pdflatex $<
	bibtex $*.aux
	pdflatex $<
	pdflatex $<

