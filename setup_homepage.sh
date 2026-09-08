#!/usr/bin/env bash
# Installs Ruby 3.2.2 via rbenv and the site's gems.
# Requires sudo. If you don't have root, see the conda/micromamba route in README.md.
set -euo pipefail

RUBY_VERSION="3.2.2"

sudo apt-get update
# System Ruby conflicts with rbenv shims; tolerate it already being absent.
sudo apt-get remove -y ruby || true
sudo apt-get install -y build-essential libssl-dev libyaml-dev libreadline-dev \
    zlib1g-dev libffi-dev libgmp-dev imagemagick

[ -d ~/.rbenv ] || git clone https://github.com/rbenv/rbenv.git ~/.rbenv
[ -d ~/.rbenv/plugins/ruby-build ] || \
    git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build

# Only append to ~/.zshrc once, so re-running stays idempotent.
grep -q 'rbenv init' ~/.zshrc 2>/dev/null || {
  echo 'export PATH="$HOME/.rbenv/bin:$HOME/.rbenv/plugins/ruby-build/bin:$PATH"' >> ~/.zshrc
  echo 'eval "$(rbenv init - zsh)"' >> ~/.zshrc
}

# Activate rbenv in *this* shell rather than `exec $SHELL`, which would
# replace the process and abandon every remaining step.
export PATH="$HOME/.rbenv/bin:$HOME/.rbenv/plugins/ruby-build/bin:$PATH"
eval "$(rbenv init - bash)"

rbenv install -s "$RUBY_VERSION"
rbenv global "$RUBY_VERSION"
ruby -v

bundle install

echo
echo "Done. Open a new shell, then run:  bundle exec jekyll serve"
