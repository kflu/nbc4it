prj=main
src=$(prj).tex
aux=$(prj).aux
DVIOUT=main.dvi

main: *.tex *.bib
	latex $(src)
	bibtex $(aux)
	latex $(src)
	latex $(src)

v:
	xdvi $(DVIOUT) &

pdf: main
	dvipdfmx $(prj).dvi

e:
	vim main.tex

clean:
	rm -f *~
	rm -f *.bbl
	rm -f *.toc
	rm -f *.aux
	rm -f *.blg
	rm -f *.log
	rm -f *.dvi
	rm -f *.pdf
	rm -f *.ps
