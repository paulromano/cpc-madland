slides = siam-madland

all: build/${slides}.pdf

images: images/spectrum-semilogy.eps

images/spectrum-semilogy.eps: images/spectrum.py
	cd images; python spectrum.py

build/${slides}.pdf: ${slides}.tex references.bib images
	mkdir -p build
	latex --output-directory=build -halt-on-error $<
	bibtex build/${slides}
	latex --output-directory=build -halt-on-error $<
	latex --output-directory=build -halt-on-error $<
	cd build; dvips ${slides}.dvi
	cd build; ps2pdf ${slides}.ps

clean:
	@rm -rf build

.PHONY: all images clean
