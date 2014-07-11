slides = siam-madland

all: build/${slides}.pdf

images: images/spectrum-semilogy.pdf

images/spectrum-semilogy.pdf: images/spectrum.py
	cd images; python spectrum.py

build/${slides}.pdf: ${slides}.tex references.bib images
	mkdir -p build
	pdflatex --output-directory=build -halt-on-error $<
	bibtex build/${slides}
	pdflatex --output-directory=build -halt-on-error $<
	pdflatex --output-directory=build -halt-on-error $<

clean:
	@rm -rf build

.PHONY: all images clean
