
# Docker and Docker Compose Installation Guide

This guide provides step-by-step instructions for installing Docker and Docker Compose on Ubuntu-based systems.

## Prerequisites

Before starting the installation, ensure you have sudo privileges on your system.

## Installation Steps

### 1. Update Package Database
```bash
sudo apt update -y
```

### 2. Install Required Packages
```bash
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
```

### 3. Add Docker's GPG Key
```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

### 4. Add Docker Repository
```bash
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```

### 5. Install Docker
```bash
sudo apt update -y
sudo apt install -y docker-ce
```

### 6. Configure Docker Service
```bash
sudo systemctl start docker
sudo systemctl enable docker
sudo systemctl status docker
```

### 7. Install Docker Compose
```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```

### 8. Set Docker Compose Permissions
```bash
sudo chmod +x /usr/local/bin/docker-compose
```

### 9. Create Symbolic Link
```bash
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
```

### 10. Add User to Docker Group
```bash
sudo usermod -aG docker $USER
```

### 11. Verify Installation
```bash
docker-compose --version
```

## Post-Installation Notes

1. After adding your user to the Docker group, you need to log out and log back in for the changes to take effect.
2. You can now start building containers using `docker-compose build`
3. All commands listed above require administrator privileges (sudo)

## Troubleshooting

If you encounter permission issues:
1. Make sure you've logged out and logged back in after adding your user to the Docker group
2. Verify Docker service is running: `sudo systemctl status docker`
3. Check if your user is in the Docker group: `groups $USER`

## Additional Resources

- [Official Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## Support

If you encounter any issues during installation, please:
1. Check the system requirements
2. Ensure all commands were executed with proper permissions
3. Verify your system's compatibility with Docker
