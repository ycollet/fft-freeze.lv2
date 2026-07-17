PREFIX ?= ~/.lv2
PLUGIN_URI = fft-freeze
BUNDLE = $(PLUGIN_URI).lv2

CFLAGS += -O2 -fPIC -Wall -Wextra -I/usr/include
CFLAGS += $(shell pkg-config --cflags lv2)
LDFLAGS += -shared $(shell pkg-config --libs lv2) -lfftw3 -lm

all: $(BUNDLE)/$(PLUGIN_URI).so

$(BUNDLE):
	mkdir -p $(BUNDLE)

$(BUNDLE)/$(PLUGIN_URI).so: $(BUNDLE) src/fft_freeze.c
	$(CC) $(CFLAGS) -o $@ src/fft_freeze.c $(LDFLAGS)

install: all
	mkdir -p $(PREFIX)/$(BUNDLE)
	cp -r $(BUNDLE)/* $(PREFIX)/$(BUNDLE)/

clean:
	rm -f $(BUNDLE)/$(PLUGIN_URI).so
