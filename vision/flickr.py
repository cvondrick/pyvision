import config
import database
import httplib
import urllib
import logging
import random
import time
import Image
import os
from xml.etree import ElementTree

log = logging.getLogger("flickr")

all_extras = "description,license,date_upload,date_taken,owner_name,"
"icon_server,original_format,last_update,geo,tags,machine_tags,o_dims,"
"views,media,path_alias,url_sq,url_t,url_s,url_m,url_o"

class Photo(object):
    """
    A photo structure that represents a photo on Flickr.

    A photo described here does not neccessarily exist on local storage. It can
    be downloaded with the download() method, which returns an PIL image.
    """

    def __init__(self, localpath, remoteurl, size, format, flickrid):
        self.localpath = localpath
        self.remoteurl = remoteurl
        self.width, self.height = size
        self.format = format
        self.flickrid = int(flickrid)

    def download(self):
        """
        Downloads the Flickr image and returns the PIL image.

        This does not write to local storage. To do so, use the save() method
        on the returned image.
        """
        data = urllib.urlopen(self.remoteurl).read()
        s = StringIO.StringIO(data)
        return Image.open(s)

    @classmethod
    def fromapi(cls, attrib):
        """
        Constructs a photo object from the Flickr API XML specification.
        """
        if "url_o" in attrib:
            url = attrib["url_o"] 
            size = attrib["width_o"], attrib["height_o"]
            format = "original"
        elif "url_l" in attrib:
            url = attrib["url_l"]
            size = attrib["width_l"], attrib["height_l"]
            format = "large"
        elif "url_m" in attrib:
            url = attrib["url_m"]
            size = attrib["width_m"], attrib["height_m"]
            format = "medium"
        elif "url_s" in attrib:
            url = attrib["url_s"]
            size = attrib["width_s"], attrib["height_s"]
            format = "small"
        else:
            raise RuntimeError("Photo does not have URL")

        return Photo(None, url, size, format, attrib["id"])

def request(method, parameters = {}):
    """
    Generic request method to the Flickr API.
    """
    apikey = random.choice(config.Flickr.api_key)
    parameters = urllib.urlencode(parameters)
    url = "/services/rest?method={0}&format=rest&api_key={1}&{2}"
    url = url.format(method, apikey, parameters)
    log.debug("Request to {0} with query {1}".format(method, url))
    conn = httplib.HTTPConnection(config.Flickr.server)
    conn.request("GET", url)
    response = ElementTree.fromstring(conn.getresponse().read())
    conn.close()
    return response

def search(tags, perpage = 100):
    """
    Builds an iterator for all the photos returned by a Flickr search. This
    function automatically jumps to the next page when a page is exhausted.
    """
    page, pages = 1, 2
    while page < pages:
        try:
            photos = request("flickr.photos.search", {
                "tags": tags,
                "page": page,
                "per_page": perpage,
                "extras": all_extras})
            photos = photos.find("photos")
            pages = int(photos.get("pages"))
            page += 1
            for photo in photos:
                yield Photo.fromapi(photo.attrib)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.exception(e)

def recent(perpage = 500):
    """
    Builds an iterator for the most recent Flickr photos.
    """
    photos = request("flickr.photos.getRecent", {
        "per_page": perpage, 
        "extras": all_extras})
    for photo in photos.getiterator("photo"):
        yield Photo.fromapi(photo.attrib)

def pascal(tags, range = (2003, 2009)):
    """
    Builds an iterator that queries images identical to the Pascal challenge.
    """
    if isinstance(tags, str):
        tags = tags.split(" ")
    perpage = 10
    while True:
        try:
            start = int(time.mktime([range[0],1,1,0,0,0,0,0,-1]))
            stop  = int(time.mktime([range[1],12,31,23,59,59,0,0,-1]))
            rtime = time.localtime(random.randint(start, stop))[0:3] + (0, 0, 0, 0, 0, -1)
            rtime = time.mktime(rtime)
            stm, etm = int(rtime), int(rtime + 86400)

            log.debug("Time range is {0} to {1}".format(time.ctime(stm), time.ctime(etm)))

            tag = random.choice(tags)

            r = request("flickr.photos.search", {
                "text": tag,
                "min_upload_date": stm,
                "max_upload_date": etm,
                "per_page": perpage,
                "page": 1})
            totpages = int(r.find("photos").get("pages"))

            if totpages == 0:
                continue

            page = random.randint(1, totpages)
            r = request("flickr.photos.search", {
                "text": tag,
                "min_upload_date": stm,
                "max_upload_date": etm,
                "per_page": perpage,
                "page": page,
                "extras": all_extras})
            photos = [x.attrib for x in r.find("photos")]
            if len(photos) > 0:
                yield Photo.fromapi(photos[random.randint(0, len(photos) - 1)])
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.exception(e)

def filtersizes(iterator, size = "medium"):
    """
    Filters sizes from the iterator and only returns images that are the
    specified minimum size.

    By default, it filters out all the small images. To download only
    originals, pass "original". Valid sizes are:
        - small
        - medium
        - large
        - original
    """
    sizes = {"small": 0, "medium": 1, "large": 2, "original": 3}
    required = sizes[size]
    for x in iterator:
        if x.format in sizes and sizes[x.format] >= required:
            yield x

def delay(iterator, wait = 1, every = 1):
    """
    Delays between every n images for the specified time. Useful so that Flickr
    doesn't disable your API key for excessive usage.
    """
    for i, photo in enumerate(iterator):
        if i > 0 and i % every == 0:
            time.sleep(wait)
        yield photo

def scrape(iterator, limit, classification, root = "store/flickr"):
    """
    Downloads up the limit of all photos in an iterator.
    """
    session = database.connect()
    total = 0
    duplicates = 0
    root = root + "/" + classification
    try:
        for photo in iterator:
            try:
                total += 1
                q = session.query(Photo)
                q = q.filter(Photo.flickrid == photo.flickrid)
                q = q.filter(Photo.classification == classification)
                if q.count() > 0:
                    duplicates += 1
                    log.info("Skipping duplicate {0}".format(photo.flickrid))
                    continue

                log.info("Downloading {0} ({1})".format(photo.flickrid, photo.format))
                filepath = "{0}/{1}/{2}.jpg".format(root, photo.flickrid % 100, photo.flickrid)
                image = photo.download()
                try:
                    image.save(filepath)
                except IOError:
                    os.makedirs(os.path.dirname(filepath))
                    image.save(filepath)

                photo.localpath = filepath
                photo.classification = classification

                session.add(photo)
                session.commit()

                limit -= 1
                if limit == 0:
                    break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.exception(e)
    finally:
        log.info("{0} / {1} were duplicates".format(duplicates, total))
        session.close()
