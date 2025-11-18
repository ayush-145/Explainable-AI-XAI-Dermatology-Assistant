# --- Stage 1: Build the React App ---
FROM node:18-alpine AS build

# Set working directory
WORKDIR /app

# Copy package.json and install dependencies
COPY package.json ./
COPY package-lock.json ./
RUN npm install

# Copy the rest of the app's source code
COPY . .

# Build the production-ready static files
RUN npm run build

# --- Stage 2: Serve with Nginx ---
FROM nginx:1.21-alpine

# Copy the built files from the 'build' stage
COPY --from=build /app/build /usr/share/nginx/html

# Tell Nginx to run
CMD ["nginx", "-g", "daemon off;"]