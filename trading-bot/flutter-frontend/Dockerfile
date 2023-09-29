# This is a more complex Dockerfile due to the specifics of Flutter
# You might not actually containerize your Flutter frontend, but here's a starting point if you do:

# Use the official Flutter SDK as the base image
FROM google/flutter

# Set the working directory
WORKDIR /app

# Install the Flutter app
COPY . .
RUN flutter pub get
RUN flutter build web

# Use a lightweight server for the built web assets
FROM nginx:alpine
COPY --from=0 /app/build/web /usr/share/nginx/html

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
