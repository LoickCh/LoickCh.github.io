### Source
Source: https://github.com/alshedivat/al-folio

### Installation

#### Local setup

To install Ruby and Bundler, simply execute the following command lines:

```bash
$ git clone git@github.com:<your-username>/<your-repo-name>.git
$ cd <your-repo-name>
$ zsh setup_homepage.sh
```

#### Local modifs

To see local modifications without deploying the webpage, execute the following command:
```bash
$ bundle exec jekyll serve
```
 If you encountered problems with images, try to install imagick:

```bash
$ sudo apt install imagemagick
```
