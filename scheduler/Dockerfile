# syntax=docker/dockerfile:1.6
FROM alpine:3.20

RUN apk add --no-cache \
    docker-cli \
    tzdata

# crond runs as root and reads crontabs from /var/spool/cron/crontabs/<user>
COPY crontab /var/spool/cron/crontabs/root
RUN chmod 0644 /var/spool/cron/crontabs/root

# Foreground cron with debug level 8 so log lines surface in `docker logs`
CMD ["crond", "-f", "-d", "8"]
