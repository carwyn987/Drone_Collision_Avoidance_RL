// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Bullet.generated.h"

UCLASS()
class AIRSIM_API ABullet : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ABullet();

	// To gain the ability to edit the above created component, these two lines enable viewing and editing anywhere
	UPROPERTY(EditAnywhere, Category = "Components")
	// The class declaration means "we will define this later so even though you don't know the type, don't worry about it"
	class UStaticMeshComponent* BulletMesh;

	UPROPERTY(EditAnywhere, Category = "Components")
	class UProjectileMovementComponent* BulletMovement;

	UPROPERTY(EditAnywhere, Category = "Counts")
	int deleteTick = -1;

	UPROPERTY(EditAnywhere, Category = "Counts")
	int curTick = 0;

	UPROPERTY(EditAnywhere, Category = "Components")
	class USphereComponent* CollisionComponent;

	UFUNCTION()
	void OnHit();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	
	
};
