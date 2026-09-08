module Jekyll
  module HideCustomBibtex
    def hideCustomBibtex(input)
      input = input.to_s
      keywords = @context.registers[:site].config['filtered_bibtex_keywords'] || []

      keywords.each do |keyword|
        # Anchor on the field name and escape the keyword. An unanchored
        # /.*keyword.*/ also matches the keyword inside values -- e.g. "code"
        # inside "Encoder-Decoder" -- and would drop the whole title line.
        input = input.gsub(/^\s*#{Regexp.escape(keyword)}\s*=.*$\n/i, '')
      end

      return input
    end
  end
end

Liquid::Template.register_filter(Jekyll::HideCustomBibtex)
