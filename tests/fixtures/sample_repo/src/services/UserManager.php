<?php

namespace App\Services;

// Global function
function formatName(string $firstName, string $lastName): string {
    return trim($firstName . ' ' . $lastName);
}

// Interface definition
interface UserInterface {
    public function getId(): int;
    public function getProfile(): ?array;
}

// Abstract class
abstract class BaseUser implements UserInterface {
    protected $id;
    
    public function getId(): int {
        return $this->id;
    }
    
    abstract protected function validate(): bool;
}

// Concrete class implementing interface and extending abstract class
class User extends BaseUser {
    private string $firstName;
    private string $lastName;
    private ?array $profile = null;

    public function __construct(int $id, string $firstName, string $lastName) {
        $this->id = $id;
        $this->firstName = $firstName;
        $this->lastName = $lastName;
    }

    public function getFullName(): string {
        return formatName($this->firstName, $this->lastName);
    }

    public function getProfile(): ?array {
        return $this->profile;
    }

    protected function validate(): bool {
        return strlen($this->firstName) > 0 && strlen($this->lastName) > 0;
    }
}

// Class with trait usage
trait LoggerTrait {
    private function log(string $message): void {
        // Logging implementation
    }
}

class UserManager {
    use LoggerTrait;

    private array $users = [];

    public function addUser(User $user): void {
        $this->users[] = $user;
        $this->log("Added user: " . $user->getFullName());
    }

    public function findUserById(int $id): ?User {
        foreach ($this->users as $user) {
            if ($user->getId() === $id) {
                return $user;
            }
        }
        return null;
    }
} 