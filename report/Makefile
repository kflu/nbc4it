prj=main
slides=slides
src=$(prj).tex
aux=$(prj).aux
DVIOUT=main.dvi

main: *.tex *.bib
	latex $(src)
	bibtex $(aux)
	latex $(src)
	latex $(src)

pdf: main
	dvipdfmx $(prj).dvi

v:
	xdvi $(DVIOUT) &

e:
	vim main.tex

s: $(slides).tex
	pdflatex $(slides).tex

sv:
	xpdf $(slides).pdf &

se:
	vim $(slides).tex

clean:
	rm -f *~
	rm -f *.bbl
	rm -f *.toc
	rm -f *.aux
	rm -f *.blg
	rm -f *.log
	rm -f *.nav
	rm -f *.out
	rm -f *.snm
	rm -f *.dvi
	rm -f *.pdf
	rm -f *.ps
