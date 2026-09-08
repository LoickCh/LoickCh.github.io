require 'feedjira'
require 'httparty'
require 'jekyll'

module ExternalPosts
  class ExternalPostsGenerator < Jekyll::Generator
    # Not `safe` -- this generator makes outbound HTTP requests.
    safe false
    priority :high

    def generate(site)
      sources = site.config['external_sources']
      return if sources.nil? || sources.empty?

      sources.each do |src|
        next if src['rss_url'].nil? || src['rss_url'].empty?
        Jekyll.logger.info "ExternalPosts:", "Fetching from #{src['name']}"

        # A dead feed or an offline machine must not abort the whole build.
        begin
          response = HTTParty.get(src['rss_url'], timeout: 10)
          raise "HTTP #{response.code}" unless response.success?
          feed = Feedjira.parse(response.body.to_s)
        rescue StandardError => e
          Jekyll.logger.warn "ExternalPosts:", "Skipping #{src['name']}: #{e.message}"
          next
        end

        seen = {}
        feed.entries.each do |e|
          # A nil date would blow up the :year interpolation in the permalink.
          next if e.title.nil? || e.title.strip.empty? || e.published.nil?

          slug = e.title.downcase.strip.gsub(' ', '-').gsub(/[^\w-]/, '')
          slug = "#{e.published.strftime('%Y-%m-%d')}-#{slug}"
          next if seen[slug]
          seen[slug] = true

          Jekyll.logger.info "ExternalPosts:", "...#{e.url}"
          path = site.in_source_dir("_posts/#{slug}.md")
          doc = Jekyll::Document.new(
            path, { :site => site, :collection => site.collections['posts'] }
          )
          doc.data['layout'] = 'post'
          doc.data['external_source'] = src['name']
          doc.data['feed_content'] = e.content || e.summary
          doc.data['title'] = e.title.to_s
          doc.data['description'] = e.summary
          doc.data['date'] = e.published
          doc.data['redirect'] = e.url
          site.collections['posts'].docs << doc
        end
      end
    end
  end
end
