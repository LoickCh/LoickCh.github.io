sudo apt-get update
sudo apt-get remove ruby
git clone https://github.com/rbenv/rbenv.git ~/.rbenv
echo 'eval "$(~/.rbenv/bin/rbenv init - zsh)"' >> ~/.zshrc

git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build
echo 'export PATH="$HOME/.rbenv/plugins/ruby-build/bin:$PATH"' >> ~/.zshrc

exec $SHELL

sudo apt-get install -y libssl-dev
rbenv install 3.1.2
rbenv global 3.1.2
ruby -v

bundle install
bundle exec jekyll serve

# Source
# https://gist.github.com/Koroeskohr/3c685f243928226f47c43080007c7e09
# https://stackoverflow.com/questions/37720892/you-dont-have-write-permissions-for-the-var-lib-gems-2-3-0-directory
